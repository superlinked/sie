from __future__ import annotations

import math
import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sie_placement.planner import CapacityExceededError, PlannerInputError, PlannerParameters, plan
from sie_placement.report import (
    Blocker,
    CapacityScope,
    ComeAliveBasis,
    IdentityStatus,
    Measurement,
    RequestReason,
    ScarcityReason,
    UnplacedReason,
    render,
)
from sie_placement.schema import (
    BootEvidence,
    BootOutcome,
    Capacity,
    Cell,
    ContainerShape,
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
    ServiceLevel,
    ServiceLevels,
    SharedQuota,
    SourceRef,
    Throughput,
    TypeCeiling,
    Warmth,
)

START = datetime(2026, 11, 2, tzinfo=UTC)
MONTH = 730.0
SCALE = MONTH / 24
L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
H100 = Cell(machine_profile="h100", gpu_product="NVIDIA-H100-80GB-HBM3")
RATES = {L4: 1.0, A100: 3.0, H100: 4.0}
CARD_MIB = {L4: 23034, A100: 81920, H100: 81559}
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
PAIR = ContainerShape(gpu_count=2, cpu_cores=8, memory_mib=32768)
IDENTITY = "org/model"
OTHER = "org/other"
PRICES = PriceList(
    currency="USD",
    machine_profiles=tuple(
        MachineProfilePrice(
            name=cell.machine_profile,
            gpu_product=cell.gpu_product,
            accelerator_per_hour=rate,
            core_per_hour=0.0,
            memory_gib_per_hour=0.0,
            gpu_memory_mib=CARD_MIB[cell],
        )
        for cell, rate in RATES.items()
    ),
    placements=(Placement(name="test", region_multiplier=1),),
)
UNRELIABLE = 1 / 3


def receipt(
    identity=IDENTITY, cell=L4, *, shape=SHAPE, concurrency=8, rps=10.0, p99=100.0, footprint=None, working=None
) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=shape,
        concurrency=concurrency,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=rps),
        latency=Percentiles(p99_ms=p99),
        footprint_mib=footprint,
        working_memory_mib=working,
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}/{shape.gpu_count}/{concurrency}/{rps}/{p99}"),
    )


def attempt(
    identity=IDENTITY,
    cell=L4,
    *,
    warmth=Warmth.COLD_WITH_BOUND,
    boot=60.0,
    wait=None,
    outcome=BootOutcome.SUCCEEDED,
    borrowed=False,
    shape=SHAPE,
    index=0,
) -> BootEvidence:
    return BootEvidence(
        identity=identity,
        cell=cell,
        container_shape=shape,
        warmth=warmth,
        outcome=outcome,
        boot_seconds=None if outcome is BootOutcome.UNPLACED else boot,
        placement_wait_s=wait,
        borrowed=borrowed,
        source=SourceRef(uri=f"boots/{identity}/{cell.machine_profile}/{warmth.value}/{index}"),
    )


def unplaced(cell=L4, *, index=0, identity=OTHER) -> BootEvidence:
    return attempt(identity, cell, outcome=BootOutcome.UNPLACED, index=index)


def scarce(cell=L4, *, placed=2, failed=1) -> list[BootEvidence]:
    rows = [attempt(OTHER, cell, boot=30.0, index=index) for index in range(placed)]
    rows.extend(unplaced(cell, index=placed + index) for index in range(failed))
    return rows


def levels(**bounds) -> ServiceLevels:
    return ServiceLevels(
        identities=tuple(
            IdentityServiceLevel(identity=identity, level=ServiceLevel(come_alive_p95_s=bound, warm_p99_ms=250))
            for identity, bound in sorted(bounds.items())
        )
    )


def one(bound) -> ServiceLevels:
    return levels(**{IDENTITY: bound})


def demand(hourly=None) -> Demand:
    return Demand(
        start=START,
        end=START + timedelta(hours=24),
        series=tuple(
            DemandSeries(
                identity=identity,
                operation=Operation.ENCODE,
                hours=tuple(
                    DemandHour(start=START + timedelta(hours=hour), requests=count)
                    for hour, count in sorted(counts.items())
                ),
            )
            for identity, counts in sorted((hourly or {}).items())
        ),
    )


def parameters(**overrides) -> PlannerParameters:
    values = {"placement": "test", "max_containers": 4, "fallback_floor": 2, "cooldown_s": 900} | overrides
    return PlannerParameters(**values)


def run(service_levels, receipts=(), boots=(), *, hourly=None, current=None, capacity=None, **overrides):
    return plan(
        service_levels, receipts, boots, PRICES, demand(hourly), current, parameters(**overrides), capacity=capacity
    )


def rendered(result) -> tuple[str, str]:
    return render(result.plan.to_dict()), render(result.report.to_dict())


def lane(name, cell=L4, *, fallbacks=(), floor=1, buffer=0) -> PlanLane:
    return PlanLane(
        name=name,
        queue_pool="default",
        bundle="default",
        cell=cell,
        fallback_cells=tuple(fallbacks),
        container_shape=SHAPE,
        concurrency=8,
        max_containers=4,
        envelope=Envelope(floor_min=floor, floor_max=floor + buffer),
    )


def plan_of(lanes, residents) -> Plan:
    pinning = {name for name, warmth in residents.values() if warmth is Warmth.PINNED}
    return Plan(
        placement="test",
        lanes=tuple(
            replace(lane, queue_pool=lane.name) if lane.name in pinning and lane.queue_pool == "default" else lane
            for lane in lanes
        ),
        identities=tuple(
            PlanIdentity(identity=identity, lane=name, warmth=warmth)
            for identity, (name, warmth) in sorted(residents.items())
        ),
    )


def test_a_cold_come_alive_adds_the_placement_wait_to_the_boot():
    result = run(one(120), [receipt()], [attempt(boot=50.0, wait=30.0)])
    chosen = result.report.identity(IDENTITY).chosen
    assert (chosen.warmth, chosen.come_alive_s, chosen.placement_wait_s) == (Warmth.COLD_WITH_BOUND, 80.0, 30.0)
    assert chosen.to_dict()["placementWaitS"] == 30.0
    assert result.report.scarcity == ()


def test_placement_waits_pool_over_every_attempt_on_a_gpu_type():
    boots = [
        attempt(boot=50.0),
        attempt(OTHER, boot=500.0, wait=100.0, index=1),
        attempt(OTHER, A100, boot=10.0, wait=900.0, index=2),
    ]
    result = run(one(120), [receipt()], boots)
    report = result.report.identity(IDENTITY)
    assert report.chosen.warmth is Warmth.PINNED
    assert report.next_cheaper.blocked_by == (Blocker(field="comeAliveP95S", bound=120, measured=150.0),)
    assert report.next_cheaper.placement_wait_s == 100.0
    (entry,) = result.report.scarcity
    assert (entry.identity, entry.reason, entry.cell, entry.warmth) == (
        IDENTITY,
        ScarcityReason.PLACEMENT_WAIT,
        L4,
        Warmth.COLD_WITH_BOUND,
    )
    assert (entry.boot_s, entry.placement_wait_s, entry.come_alive_bound_s, entry.monthly_cost) == (50.0, 100.0, 120, 0)
    assert entry.to_dict()["reason"] == "placement-wait"


@pytest.mark.parametrize(("fast", "come_alive"), [(17, None), (18, 60.0), (36, None), (37, 60.0)])
def test_the_placement_wait_takes_the_95th_percentile_from_twenty_attempts(fast, come_alive):
    boots = [attempt(boot=50.0)]
    boots.extend(attempt(OTHER, boot=1.0, wait=10.0, index=index) for index in range(fast))
    boots.extend(attempt(OTHER, boot=1.0, wait=200.0, index=100 + index) for index in range(1 if fast < 30 else 2))
    chosen = run(one(120), [receipt()], boots).report.identity(IDENTITY).chosen
    if come_alive is None:
        assert chosen.warmth is Warmth.PINNED
    else:
        assert (chosen.warmth, chosen.come_alive_s) == (Warmth.COLD_WITH_BOUND, come_alive)


@pytest.mark.parametrize(("placed", "selected"), [(17, False), (18, True)])
def test_an_unplaced_attempt_ranks_above_every_placed_one(placed, selected):
    boots = [attempt(boot=50.0, wait=10.0)]
    boots.extend(attempt(OTHER, boot=1.0, wait=10.0, index=index) for index in range(placed))
    boots.append(unplaced(index=99))
    result = run(one(120), [receipt()], boots)
    chosen = result.report.identity(IDENTITY).chosen
    if selected:
        assert (chosen.warmth, chosen.come_alive_s) == (Warmth.COLD_WITH_BOUND, 60.0)
        assert result.report.scarcity == ()
        return
    assert chosen.warmth is Warmth.PINNED
    (request,) = [request for request in result.report.measurement_requests if request.cell == L4]
    assert (request.measure, request.warmth, request.reason) == (
        (Measurement.BOOT_EVIDENCE,),
        Warmth.COLD_WITH_BOUND,
        RequestReason.UNPLACED,
    )
    assert request.to_dict()["reason"] == "unplaced"
    (entry,) = result.report.scarcity
    assert (entry.reason, entry.boot_s, entry.placement_wait_s) == (ScarcityReason.UNPLACED, 50.0, None)
    assert entry.placement_failure_rate == pytest.approx(1 / (placed + 2))


@pytest.mark.parametrize(("placed", "selected"), [(17, False), (18, True), (100, True)])
def test_a_placed_attempt_without_a_placement_wait_ranks_as_one_that_waited_none(placed, selected):
    boots = [attempt(boot=50.0)]
    boots.extend(attempt(OTHER, boot=1.0, wait=0.0 if index % 2 else None, index=index) for index in range(placed))
    boots.append(unplaced(index=999))
    result = run(one(120), [receipt()], boots)
    chosen = result.report.identity(IDENTITY).chosen
    if not selected:
        assert chosen.warmth is Warmth.PINNED
        (entry,) = result.report.scarcity
        assert entry.reason is ScarcityReason.UNPLACED
        return
    assert (chosen.warmth, chosen.come_alive_s, chosen.placement_wait_s) == (Warmth.COLD_WITH_BOUND, 50.0, 0.0)
    assert result.report.scarcity == ()
    assert not [request for request in result.report.measurement_requests if request.cell == L4]


def test_placed_attempts_without_a_placement_wait_count_toward_the_failure_rate_and_the_percentile_alike():
    boots = [attempt(boot=50.0), *(attempt(OTHER, boot=1.0, index=index) for index in range(99))]
    boots.append(unplaced(index=999))
    result = run(one(120), [receipt()], boots)
    assert result.report.identity(IDENTITY).chosen.come_alive_s == 50.0
    boots = [*boots, *(unplaced(index=1000 + index) for index in range(5))]
    result = run(one(0), [receipt(), receipt(cell=A100)], boots)
    assert result.report.identity(IDENTITY).chosen.cell == A100
    (entry,) = result.report.scarcity
    assert (entry.reason, entry.placement_failure_rate) == (ScarcityReason.UNRELIABLE_TYPE, pytest.approx(6 / 106))


def test_an_attempt_without_a_placement_wait_leaves_its_boot_to_stand_for_the_come_alive():
    result = run(one(120), [receipt()], [attempt(boot=95.0)])
    chosen = result.report.identity(IDENTITY).chosen
    assert (chosen.come_alive_s, chosen.placement_wait_s) == (95.0, None)
    assert "placementWaitS" not in chosen.to_dict()


def test_own_attempts_on_a_type_replace_borrowed_ones():
    boots = [attempt(OTHER, boot=1.0, wait=200.0, borrowed=True, index=1), attempt(boot=50.0)]
    chosen = run(one(120), [receipt()], boots).report.identity(IDENTITY).chosen
    assert (chosen.come_alive_s, chosen.come_alive_basis) == (50.0, ComeAliveBasis.BOOT_EVIDENCE)
    borrowed = [attempt(boot=50.0, wait=20.0, borrowed=True)]
    chosen = run(one(120), [receipt()], borrowed).report.identity(IDENTITY).chosen
    assert (chosen.come_alive_s, chosen.come_alive_basis) == (70.0, ComeAliveBasis.BORROWED_BOOT_EVIDENCE)


def test_an_unplaced_attempt_of_the_identity_withdraws_its_cold_cell_until_it_boots():
    result = run(one(120), [receipt()], [unplaced(identity=IDENTITY)])
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    (request,) = [request for request in result.report.measurement_requests if request.cell == L4]
    assert request.reason is RequestReason.UNPLACED
    assert result.report.scarcity == ()


def test_a_snapshot_restore_waits_for_placement_too():
    boots = [attempt(warmth=Warmth.SNAPSHOT_COLD, boot=5.0, wait=30.0)]
    result = run(one(20), [receipt()], boots)
    report = result.report.identity(IDENTITY)
    assert report.chosen.warmth is Warmth.PINNED
    assert (report.next_cheaper.warmth, report.next_cheaper.come_alive_s) == (Warmth.SNAPSHOT_COLD, 35.0)
    assert [(entry.warmth, entry.reason) for entry in result.report.scarcity] == [
        (Warmth.SNAPSHOT_COLD, ScarcityReason.PLACEMENT_WAIT)
    ]


def test_a_lazy_resident_load_waits_for_no_placement():
    current = plan_of(
        [lane("own"), lane("shared")], {OTHER: ("shared", Warmth.PINNED), IDENTITY: ("own", Warmth.PINNED)}
    )
    service_levels = levels(**{OTHER: 0, IDENTITY: 60})
    boots = [attempt(warmth=Warmth.LAZY_RESIDENT, boot=20.0), attempt(OTHER, boot=1.0, wait=500.0, index=1)]
    receipts = [receipt(OTHER, footprint=4000, working=1000), receipt(footprint=2000, working=1000)]
    result = run(service_levels, receipts, boots, current=current)
    chosen = result.report.identity(IDENTITY).chosen
    assert (chosen.warmth, chosen.lane, chosen.come_alive_s, chosen.placement_wait_s) == (
        Warmth.LAZY_RESIDENT,
        "shared",
        20.0,
        None,
    )


@pytest.mark.parametrize(("wait", "under"), [(30.0, MONTH), (70.0, 0.0)])
def test_a_lever_under_its_bound_leaves_room_for_the_placement_wait(wait, under):
    result = run(one(60), [receipt()], [attempt(boot=50.0, wait=wait)])
    (lever,) = result.report.come_alive_levers
    assert (lever.come_alive_s, lever.placement_wait_s, lever.permitted_bound_s) == (50.0 + wait, wait, 50 + wait)
    assert lever.saving_at_permitted_bound == pytest.approx(MONTH)
    assert lever.saving_under_bound == pytest.approx(under)
    assert lever.to_dict()["placementWaitS"] == wait


def test_an_unreliable_type_is_not_chosen_under_a_bound_the_hold_budget_fits():
    result = run(one(0), [receipt(), receipt(cell=A100)], scarce())
    report = result.report.identity(IDENTITY)
    assert (report.chosen.cell, report.chosen.warmth) == (A100, Warmth.PINNED)
    assert report.chosen.monthly_cost == pytest.approx(3 * MONTH)
    assert report.next_cheaper.cell == L4
    assert report.next_cheaper.blocked_by == (
        Blocker(field="placementFailureRate", bound=0.05, measured=pytest.approx(UNRELIABLE)),
    )
    assert report.next_cheaper.saving == pytest.approx(2 * MONTH)
    (entry,) = result.report.scarcity
    assert (entry.reason, entry.cell, entry.warmth, entry.monthly_cost) == (
        ScarcityReason.UNRELIABLE_TYPE,
        L4,
        Warmth.PINNED,
        pytest.approx(MONTH),
    )
    assert entry.to_dict()["placementFailureRate"] == 0.3333
    assert result.report.to_dict()["identities"][0]["nextCheaper"]["blockedBy"][0]["field"] == "placementFailureRate"


def test_loosening_a_bound_above_the_hold_budget_admits_an_unreliable_type():
    report = run(one(0), [receipt(), receipt(cell=A100)], scarce(), hold_budget_s=20).report.identity(IDENTITY)
    loosened = run(one(21), [receipt(), receipt(cell=A100)], scarce(), hold_budget_s=20).report
    assert report.chosen.cell == A100
    assert loosened.identity(IDENTITY).chosen.cell == L4
    assert report.next_cheaper.saving == pytest.approx(2 * MONTH)


@pytest.mark.parametrize(("hold", "cell"), [(20, A100), (19, L4), (0, L4)])
def test_the_hold_budget_decides_which_bounds_avoid_unreliable_types(hold, cell):
    result = run(one(20), [receipt(), receipt(cell=A100)], scarce(), hold_budget_s=hold)
    assert result.report.identity(IDENTITY).chosen.cell == cell
    assert bool(result.report.scarcity) is (cell == A100)


def test_an_undeclared_come_alive_bound_may_use_an_unreliable_type():
    result = run(one(None), [receipt(), receipt(cell=A100)], scarce(), hold_budget_s=600)
    assert result.report.identity(IDENTITY).chosen.cell == L4


def test_a_placement_failure_rate_of_five_percent_is_reliable():
    boots = scarce(placed=19, failed=1)
    assert run(one(0), [receipt(), receipt(cell=A100)], boots).report.identity(IDENTITY).chosen.cell == L4
    boots = scarce(placed=18, failed=1)
    assert run(one(0), [receipt(), receipt(cell=A100)], boots).report.identity(IDENTITY).chosen.cell == A100


def test_the_failure_rate_counts_the_environments_own_attempts_over_borrowed_ones():
    borrowed = [replace(row, borrowed=True) for row in scarce(placed=0, failed=3)]
    own = [attempt(OTHER, boot=30.0, index=10)]
    result = run(one(0), [receipt(), receipt(cell=A100)], [*borrowed, *own])
    assert result.report.identity(IDENTITY).chosen.cell == L4
    result = run(one(0), [receipt(), receipt(cell=A100)], borrowed)
    assert result.report.identity(IDENTITY).chosen.cell == A100


def test_an_identity_whose_only_types_are_unreliable_is_unplaced_and_says_why():
    result = run(one(0), [receipt()], scarce())
    report = result.report.identity(IDENTITY)
    assert (report.status, report.unplaced_reason) == (IdentityStatus.UNPLACED, UnplacedReason.PLACEMENT_UNRELIABLE)
    assert report.to_dict()["unplacedReason"] == "placement-unreliable"
    assert result.plan.identities == ()


def test_unreliable_types_are_not_requested_for_a_bound_the_hold_budget_fits():
    result = run(one(0), [receipt(cell=H100)], scarce())
    assert [request.cell for request in result.report.measurement_requests] == [A100]
    result = run(one(600), [receipt(cell=H100)], scarce())
    assert sorted(request.cell.machine_profile for request in result.report.measurement_requests) == [
        "a100-80gb",
        "h100",
        "l4",
    ]


def test_a_current_placement_on_an_unreliable_type_is_known_to_miss_its_bound():
    current = plan_of([lane("own")], {IDENTITY: ("own", Warmth.PINNED)})
    moved = run(one(0), [receipt(), receipt(cell=A100)], scarce(), current=current)
    assert moved.report.identity(IDENTITY).chosen.cell == A100
    assert [entry.lane for entry in moved.plan.identities] == ["org-model-a100-80gb"]
    kept = run(one(0), [receipt()], scarce(), current=current)
    report = kept.report.identity(IDENTITY)
    assert (report.status, report.lane, report.unplaced_reason) == (
        IdentityStatus.KEPT,
        "own",
        UnplacedReason.PLACEMENT_UNRELIABLE,
    )
    assert report.blocked_by == (Blocker(field="placementFailureRate", bound=0.05, measured=pytest.approx(UNRELIABLE)),)
    assert kept.plan == current


def test_a_candidate_list_names_a_second_type_when_every_single_type_is_unreliable():
    boots = [*scarce(L4), *scarce(A100)]
    receipts = [receipt(), receipt(cell=A100)]
    result = run(one(0), receipts, boots)
    chosen = result.report.identity(IDENTITY).chosen
    assert (chosen.cell, chosen.fallback_cells, chosen.warmth) == (L4, (A100,), Warmth.PINNED)
    assert chosen.monthly_cost == pytest.approx(3 * MONTH)
    assert chosen.to_dict()["fallbackCells"] == [A100.to_dict()]
    (planned,) = result.plan.lanes
    assert (planned.name, planned.cell, planned.fallback_cells) == ("org-model-l4-a100-80gb", L4, (A100,))
    assert result.report.monthly_cost == pytest.approx(3 * MONTH)
    assert Plan.from_dict(result.plan.to_dict()) == result.plan
    again = run(one(0), receipts, boots, current=result.plan)
    assert again.plan == result.plan
    assert again.report.unchanged


def test_a_candidate_list_is_not_offered_where_a_reliable_type_is_cheaper_or_as_cheap():
    result = run(one(0), [receipt(), receipt(cell=A100)], scarce(L4))
    assert result.report.identity(IDENTITY).chosen.fallback_cells == ()
    assert result.plan.lanes[0].fallback_cells == ()


@pytest.mark.parametrize(
    "second",
    [
        receipt(cell=A100, p99=300.0),
        receipt(cell=A100, concurrency=16),
        receipt(cell=A100, shape=PAIR),
    ],
)
def test_a_list_member_meets_the_warm_bound_at_the_lane_shape_and_concurrency(second):
    result = run(one(0), [receipt(), second], [*scarce(L4), *scarce(A100)])
    assert result.report.identity(IDENTITY).status is IdentityStatus.UNPLACED


def test_a_candidate_list_is_priced_at_its_dearest_member_and_its_slowest_receipt():
    receipts = [receipt(rps=10.0), receipt(cell=A100, rps=5.0)]
    result = run(one(0), receipts, [*scarce(L4), *scarce(A100)], hourly={IDENTITY: {0: 100000}})
    report = result.report.identity(IDENTITY)
    above = 3 * SCALE
    assert report.chosen.monthly_cost == pytest.approx(3 * (MONTH + above))
    assert report.monthly_cost == pytest.approx(3 * (MONTH + above))
    assert report.peak_containers == 6
    (lane_report,) = result.report.lanes
    assert lane_report.hourly_cost == 3.0


def test_a_current_candidate_list_stays_when_a_plain_type_costs_the_same():
    current = plan_of([lane("listed", fallbacks=(A100,))], {IDENTITY: ("listed", Warmth.PINNED)})
    result = run(one(0), [receipt(), receipt(cell=A100)], scarce(L4), current=current)
    assert result.plan == current
    assert result.report.unchanged
    assert result.report.identity(IDENTITY).chosen.fallback_cells == (A100,)


def test_a_lane_with_fallback_cells_hosts_no_lazy_resident():
    current = plan_of(
        [lane("own"), lane("listed", fallbacks=(A100,))],
        {OTHER: ("listed", Warmth.PINNED), IDENTITY: ("own", Warmth.PINNED)},
    )
    service_levels = levels(**{OTHER: 0, IDENTITY: 60})
    boots = [attempt(warmth=Warmth.LAZY_RESIDENT, boot=20.0), *scarce(L4)]
    result = run(service_levels, [receipt(OTHER), receipt(OTHER, A100), receipt()], boots, current=current)
    assert {entry.identity: entry.lane for entry in result.plan.identities}[OTHER] == "listed"
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    assert all(request.warmth is not Warmth.LAZY_RESIDENT for request in result.report.measurement_requests)


def two_lanes(**overrides):
    receipts = [receipt("org/a"), receipt("org/b", A100)]
    return run(levels(**{"org/a": 0, "org/b": 0}), receipts, buffer_containers=1, **overrides)


def test_capacity_reports_what_the_envelope_maxima_and_container_ceilings_hold_against_each_ceiling():
    capacity = Capacity(
        total=4,
        types=(
            TypeCeiling(gpu_product="NVIDIA-L4", accelerators=2),
            TypeCeiling(gpu_product=A100.gpu_product, accelerators=2),
        ),
        shared_quota=SharedQuota(accelerators=6, held_elsewhere=2),
    )
    report = two_lanes(capacity=capacity, max_containers=2).report
    assert [
        (use.scope, use.gpu_product, use.ceiling, use.accelerators, use.peak_accelerators, use.lanes)
        for use in report.capacity
    ] == [
        (CapacityScope.TOTAL, None, 4, 4, 4, ("org-a-l4", "org-b-a100-80gb")),
        (CapacityScope.TYPE, A100.gpu_product, 2, 2, 2, ("org-b-a100-80gb",)),
        (CapacityScope.TYPE, "NVIDIA-L4", 2, 2, 2, ("org-a-l4",)),
        (CapacityScope.SHARED_QUOTA, None, 6, 6, 6, ("org-a-l4", "org-b-a100-80gb")),
    ]
    assert not any(use.exceeded for use in report.capacity)
    assert report.to_dict()["capacity"][0] == {
        "scope": "total",
        "ceiling": 4,
        "accelerators": 4,
        "peakAccelerators": 4,
        "lanes": ["org-a-l4", "org-b-a100-80gb"],
        "exceeded": False,
    }
    assert report.to_dict()["capacity"][3] == {
        "scope": "shared-quota",
        "ceiling": 6,
        "accelerators": 6,
        "peakAccelerators": 6,
        "heldElsewhere": 2,
        "lanes": ["org-a-l4", "org-b-a100-80gb"],
        "exceeded": False,
    }


@pytest.mark.parametrize(
    ("capacity", "message"),
    [
        (
            Capacity(types=(TypeCeiling(gpu_product="NVIDIA-L4", accelerators=1),)),
            "the container ceilings of org-a-l4 hold 4 NVIDIA-L4 accelerators, above the type ceiling of 1",
        ),
        (
            Capacity(total=3),
            "the container ceilings of org-a-l4, org-b-a100-80gb hold 8 accelerators, above the total ceiling of 3",
        ),
        (
            Capacity(total=4),
            "the container ceilings of org-a-l4, org-b-a100-80gb hold 8 accelerators, above the total ceiling of 4",
        ),
        (
            Capacity(shared_quota=SharedQuota(accelerators=6, held_elsewhere=3)),
            "the container ceilings of org-a-l4, org-b-a100-80gb hold 8 accelerators, "
            "which with 3 held elsewhere exceed the shared quota of 6",
        ),
        (
            Capacity(shared_quota=SharedQuota(accelerators=10, held_elsewhere=3)),
            "the container ceilings of org-a-l4, org-b-a100-80gb hold 8 accelerators, "
            "which with 3 held elsewhere exceed the shared quota of 10",
        ),
    ],
)
def test_a_plan_whose_lanes_exceed_a_ceiling_is_refused_and_says_why(capacity, message):
    with pytest.raises(CapacityExceededError) as caught:
        two_lanes(capacity=capacity)
    assert isinstance(caught.value, PlannerInputError)
    assert str(caught.value) == message
    (exceeded,) = caught.value.exceeded
    assert exceeded.exceeded
    refused = caught.value.result
    assert [lane.name for lane in refused.plan.lanes] == ["org-a-l4", "org-b-a100-80gb"]
    assert [use["exceeded"] for use in refused.report.to_dict()["capacity"]] == [True]


def test_every_exceeded_ceiling_is_named():
    capacity = Capacity(total=3, types=(TypeCeiling(gpu_product="NVIDIA-L4", accelerators=1),))
    with pytest.raises(CapacityExceededError) as caught:
        two_lanes(capacity=capacity)
    assert [use.scope for use in caught.value.exceeded] == [CapacityScope.TOTAL, CapacityScope.TYPE]
    assert str(caught.value).count("the container ceilings") == 2


def test_the_container_ceilings_bind_where_the_envelope_maxima_fit():
    capacity = Capacity(
        total=4,
        types=(TypeCeiling(gpu_product="NVIDIA-L4", accelerators=2),),
        shared_quota=SharedQuota(accelerators=7, held_elsewhere=3),
    )
    with pytest.raises(CapacityExceededError) as caught:
        two_lanes(capacity=capacity)
    assert [(use.scope, use.accelerators, use.peak_accelerators) for use in caught.value.exceeded] == [
        (CapacityScope.TOTAL, 4, 8),
        (CapacityScope.TYPE, 2, 4),
        (CapacityScope.SHARED_QUOTA, 7, 11),
    ]
    result = two_lanes(capacity=capacity, max_containers=2)
    assert [planned.max_containers for planned in result.plan.lanes] == [2, 2]


def test_a_refused_plan_keeps_every_container_ceiling_it_would_need():
    capacity = Capacity(total=5, shared_quota=SharedQuota(accelerators=5))
    with pytest.raises(CapacityExceededError) as caught:
        two_lanes(capacity=capacity)
    refused = caught.value.result.plan
    assert [(planned.name, planned.max_containers) for planned in refused.lanes] == [
        ("org-a-l4", 4),
        ("org-b-a100-80gb", 4),
    ]
    assert refused == two_lanes().plan
    for use in caught.value.exceeded:
        assert use.lanes == ("org-a-l4", "org-b-a100-80gb")
        assert "org-a-l4, org-b-a100-80gb hold 8 accelerators" in use.describe()


def test_a_candidate_list_lane_counts_against_every_member_type():
    capacity = Capacity(
        types=(
            TypeCeiling(gpu_product="NVIDIA-L4", accelerators=1),
            TypeCeiling(gpu_product=A100.gpu_product, accelerators=1),
        )
    )
    boots = [*scarce(L4), *scarce(A100)]
    result = run(one(0), [receipt(), receipt(cell=A100)], boots, capacity=capacity, max_containers=1)
    assert [(use.accelerators, use.peak_accelerators) for use in result.report.capacity] == [(1, 1), (1, 1)]
    capacity = Capacity(types=(TypeCeiling(gpu_product=A100.gpu_product, accelerators=0),))
    with pytest.raises(CapacityExceededError, match="hold 1 NVIDIA-A100-SXM4-80GB accelerators"):
        run(one(0), [receipt(), receipt(cell=A100)], boots, capacity=capacity, max_containers=1)


def test_each_container_holds_its_gpu_count():
    result = run(one(0), [receipt(shape=PAIR)], capacity=Capacity(total=2), buffer_containers=0, max_containers=1)
    assert [(use.accelerators, use.peak_accelerators) for use in result.report.capacity] == [(2, 2)]
    with pytest.raises(CapacityExceededError, match="hold 2 accelerators"):
        run(one(0), [receipt(shape=PAIR)], capacity=Capacity(total=1), max_containers=1)


def test_a_zero_floor_lane_holds_no_envelope_maximum_but_holds_its_container_ceiling():
    result = run(one(600), [receipt()], [attempt(boot=60.0)], capacity=Capacity(total=4))
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.COLD_WITH_BOUND
    assert [(use.accelerators, use.peak_accelerators, use.lanes) for use in result.report.capacity] == [
        (0, 4, ("org-model-l4",))
    ]
    with pytest.raises(CapacityExceededError, match="the container ceilings of org-model-l4 hold 4 accelerators"):
        run(one(600), [receipt()], [attempt(boot=60.0)], capacity=Capacity(total=3))


def test_a_loosening_whose_plan_the_ceilings_would_refuse_saves_nothing():
    receipts, boots = [receipt()], [attempt(boot=240.0)]
    current = plan_of(
        [replace(lane("org-model-l4", buffer=1), max_containers=2)], {IDENTITY: ("org-model-l4", Warmth.PINNED)}
    )
    free = run(one(60), receipts, boots, buffer_containers=1, fallback_floor=3, current=current)
    assert free.plan == current
    assert [lever.permitted_bound_s for lever in free.report.come_alive_levers] == [240]
    capacity = Capacity(shared_quota=SharedQuota(accelerators=2))
    bounded = run(one(60), receipts, boots, buffer_containers=1, fallback_floor=3, capacity=capacity, current=current)
    assert bounded.plan == free.plan
    assert bounded.report.come_alive_levers == ()
    assert bounded.report.identity(IDENTITY).next_cheaper.saving == 0
    with pytest.raises(CapacityExceededError, match="the container ceilings of org-model-l4 hold 4 accelerators"):
        run(one(240), receipts, boots, buffer_containers=1, fallback_floor=3, capacity=capacity, current=free.plan)


def test_without_capacity_the_report_lists_no_ceiling():
    report = two_lanes().report
    assert report.capacity == ()
    assert report.to_dict()["capacity"] == []


CELLS = (L4, A100, H100)
FIELDS = {"comeAliveP95S": "come_alive_p95_s", "warmP99Ms": "warm_p99_ms"}


@pytest.mark.parametrize("seed", range(120))
def test_random_estates_respect_placement_evidence_and_capacity(seed):
    rng = random.Random(seed)  # noqa: S311 -- reproducible test data
    identities = [f"org/model-{index}" for index in range(rng.randint(1, 4))]
    service_levels = ServiceLevels(
        identities=tuple(
            IdentityServiceLevel(
                identity=identity,
                level=ServiceLevel(
                    come_alive_p95_s=rng.choice([None, 0, 10, 30, 120, 600]),
                    warm_p99_ms=rng.choice([150, 250, 400]),
                ),
            )
            for identity in identities
        )
    )
    receipts = [
        receipt(
            identity, cell, concurrency=rng.choice([4, 8]), rps=rng.choice([0.5, 5.0]), p99=rng.choice([100.0, 300.0])
        )
        for identity in identities
        for cell in CELLS
        if rng.random() < 0.6
    ]
    boots = []
    for identity in identities:
        for cell in CELLS:
            for warmth in (Warmth.COLD_WITH_BOUND, Warmth.SNAPSHOT_COLD):
                for index in range(rng.randint(0, 3)):
                    outcome = rng.choice(
                        [BootOutcome.SUCCEEDED, BootOutcome.SUCCEEDED, BootOutcome.FAILED, BootOutcome.UNPLACED]
                    )
                    boots.append(
                        attempt(
                            identity,
                            cell,
                            warmth=warmth,
                            boot=rng.choice([5.0, 20.0, 90.0]),
                            wait=rng.choice([None, 0.0, 15.0, 120.0]),
                            outcome=outcome,
                            borrowed=rng.random() < 0.2,
                            index=index,
                        )
                    )
    for index in range(rng.randint(0, 6)):
        boots.append(unplaced(rng.choice(CELLS), index=index))
    hold = rng.choice([0, 20, 60])
    capacity = rng.choice(
        [
            None,
            Capacity(total=rng.randint(0, 8)),
            Capacity(
                types=tuple(TypeCeiling(gpu_product=cell.gpu_product, accelerators=rng.randint(0, 4)) for cell in CELLS)
            ),
            Capacity(shared_quota=SharedQuota(accelerators=8, held_elsewhere=rng.randint(0, 8))),
        ]
    )
    overrides = {"hold_budget_s": hold, "buffer_containers": rng.randint(0, 1)}
    hourly = {
        identity: {hour: rng.choice([1, 500]) for hour in rng.sample(range(24), rng.randint(0, 3))}
        for identity in identities
    }

    def attempt_plan(current=None, order=False, given_levels=service_levels):
        given_receipts, given_boots = list(receipts), list(boots)
        if order:
            rng.shuffle(given_receipts)
            given_boots.reverse()
        try:
            return run(
                given_levels,
                given_receipts,
                given_boots,
                hourly=hourly,
                current=current,
                capacity=capacity,
                **overrides,
            ), None
        except CapacityExceededError as error:
            return error.result, error

    def relaxed(identity, blockers):
        bounds = {}
        for blocker in blockers:
            if blocker.field == "placementFailureRate":
                name, value = "come_alive_p95_s", hold + 1
            elif blocker.measured is not None:
                name, value = FIELDS[blocker.field], max(1, math.ceil(blocker.measured))
            else:
                continue
            bounds[name] = max(value, bounds.get(name, 0))
        return ServiceLevels(
            identities=tuple(
                replace(entry, level=replace(entry.level, **bounds)) if entry.identity == identity else entry
                for entry in service_levels.identities
            )
        )

    def fall(identity, blockers):
        again, again_error = attempt_plan(current=result.plan, given_levels=relaxed(identity, blockers))
        return 0.0 if again_error is not None else max(0.0, result.report.monthly_cost - again.report.monthly_cost)

    result, error = attempt_plan()
    attempts: dict[str, list[BootEvidence]] = {}
    for row in boots:
        if row.warmth in (Warmth.COLD_WITH_BOUND, Warmth.SNAPSHOT_COLD):
            attempts.setdefault(row.cell.gpu_product, []).append(row)

    def failure_rate(product):
        rows = attempts.get(product, [])
        own = [row for row in rows if not row.borrowed] or rows
        return sum(row.outcome is BootOutcome.UNPLACED for row in own) / len(own) if own else 0.0

    by_uri = {document.source.uri: document for document in receipts}
    levels_by_identity = {entry.identity: entry.level for entry in service_levels.identities}
    for entry in result.report.identities:
        if entry.chosen is None:
            continue
        chosen = entry.chosen
        level = levels_by_identity[entry.identity]
        assert chosen.blocked_by == ()
        tight = level.come_alive_p95_s is not None and level.come_alive_p95_s <= hold
        if tight and not chosen.fallback_cells:
            assert failure_rate(chosen.cell.gpu_product) <= 0.05
        if chosen.fallback_cells:
            assert chosen.warmth is Warmth.PINNED
            assert tight
            assert failure_rate(chosen.cell.gpu_product) > 0.05
            (second,) = chosen.fallback_cells
            assert second.gpu_product != chosen.cell.gpu_product
            assert RATES[second] >= RATES[chosen.cell]
            assert any(
                document.identity == entry.identity
                and document.cell == second
                and document.concurrency == chosen.concurrency
                and document.latency.p99_ms <= level.warm_p99_ms
                for document in receipts
            )
        assert by_uri[chosen.receipt.uri].latency.p99_ms <= level.warm_p99_ms
        if chosen.warmth is not Warmth.PINNED:
            assert level.come_alive_p95_s is None or chosen.come_alive_s <= level.come_alive_p95_s
            if chosen.placement_wait_s is not None:
                assert chosen.come_alive_s >= chosen.placement_wait_s
    for scarcity in result.report.scarcity:
        chosen = result.report.identity(scarcity.identity).chosen
        if chosen is not None:
            assert scarcity.monthly_cost < chosen.monthly_cost
        if scarcity.reason is ScarcityReason.UNRELIABLE_TYPE:
            assert scarcity.placement_failure_rate > 0.05
    for use in result.report.capacity:
        lanes = [
            planned
            for planned in result.plan.lanes
            if use.scope is not CapacityScope.TYPE or use.gpu_product in {cell.gpu_product for cell in planned.cells}
        ]
        held = sum(planned.envelope.floor_max * planned.container_shape.gpu_count for planned in lanes)
        peak = sum(planned.max_containers * planned.container_shape.gpu_count for planned in lanes)
        elsewhere = use.held_elsewhere or 0
        assert use.accelerators == held + elsewhere
        assert use.peak_accelerators == peak + elsewhere
        assert use.exceeded is (peak + elsewhere > use.ceiling)
        assert use.lanes == tuple(planned.name for planned in lanes)
    assert (error is not None) is any(use.exceeded for use in result.report.capacity)
    if error is not None:
        assert error.exceeded == tuple(use for use in result.report.capacity if use.exceeded)
    reordered, _ = attempt_plan(order=True)
    assert rendered(reordered) == rendered(result)
    if error is None:
        for entry in result.report.identities:
            if entry.next_cheaper is not None:
                assert entry.next_cheaper.saving == pytest.approx(
                    fall(entry.identity, entry.next_cheaper.blocked_by), abs=0.01
                )
        for lever in result.report.come_alive_levers:
            measured = Blocker(field="comeAliveP95S", bound=None, measured=lever.permitted_bound_s)
            assert lever.saving_at_permitted_bound == pytest.approx(fall(lever.identity, [measured]), abs=0.01)
        again, again_error = attempt_plan(current=result.plan)
        assert again_error is None
        assert again.plan == result.plan
        assert again.report.unchanged
