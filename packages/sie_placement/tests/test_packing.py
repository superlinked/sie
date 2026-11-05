from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sie_placement.planner import PlannerParameters, plan
from sie_placement.report import IdentityStatus, Measurement, RequestReason, UnplacedReason, render
from sie_placement.schema import (
    BootEvidence,
    BootOutcome,
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
    SourceRef,
    Throughput,
    Warmth,
)

START = datetime(2026, 11, 2, tzinfo=UTC)
MONTH = 730.0
L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
L40S = Cell(machine_profile="l40s", gpu_product="NVIDIA-L40S")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
RATES = {L4: 1.0, L40S: 2.0, A100: 3.0}
CARD_MIB = {L4: 23034, L40S: 46068, A100: 81920}
LIMIT = 23034 * 0.8
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
PAIR = ContainerShape(gpu_count=2, cpu_cores=8, memory_mib=32768)
HOST = "org/host"
GUEST = "org/guest"


def prices(*, cards=True) -> PriceList:
    return PriceList(
        currency="USD",
        machine_profiles=tuple(
            MachineProfilePrice(
                name=cell.machine_profile,
                gpu_product=cell.gpu_product,
                accelerator_per_hour=rate,
                core_per_hour=0.0,
                memory_gib_per_hour=0.0,
                gpu_memory_mib=CARD_MIB[cell] if cards else None,
            )
            for cell, rate in RATES.items()
        ),
        placements=(Placement(name="test", region_multiplier=1),),
    )


def receipt(
    identity,
    cell=L4,
    *,
    beside=(),
    footprint=None,
    working=None,
    p99=100.0,
    rps=10.0,
    shape=SHAPE,
    concurrency=8,
) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=shape,
        concurrency=concurrency,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=rps),
        latency=Percentiles(p99_ms=p99),
        co_residents=tuple(beside),
        footprint_mib=footprint,
        working_memory_mib=working,
        source=SourceRef(
            uri=f"receipts/{identity}/{cell.machine_profile}/{shape.gpu_count}/{concurrency}/{'+'.join(beside)}/{p99}"
        ),
    )


def generation_receipt(identity, cell=L4, *, footprint=None, working=None) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        concurrency=8,
        request_shape=RequestShape(operation=Operation.GENERATE, items_per_request=1, output_tokens=128),
        throughput=Throughput(requests_per_second=2.0, output_tokens_per_second=256.0),
        ttft=Percentiles(p99_ms=500.0),
        itl=Percentiles(p99_ms=40.0),
        footprint_mib=footprint,
        working_memory_mib=working,
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}/generate"),
    )


def load(identity, cell=L4, seconds=20.0, *, footprint=None, outcome=BootOutcome.SUCCEEDED) -> BootEvidence:
    return BootEvidence(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        warmth=Warmth.LAZY_RESIDENT,
        outcome=outcome,
        boot_seconds=seconds,
        footprint_mib=footprint,
        source=SourceRef(uri=f"loads/{identity}/{cell.machine_profile}"),
    )


def cold(identity, cell=L4, seconds=30.0) -> BootEvidence:
    return BootEvidence(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        warmth=Warmth.COLD_WITH_BOUND,
        outcome=BootOutcome.SUCCEEDED,
        boot_seconds=seconds,
        source=SourceRef(uri=f"boots/{identity}/{cell.machine_profile}"),
    )


def levels(**bounds) -> ServiceLevels:
    return ServiceLevels(
        identities=tuple(
            IdentityServiceLevel(identity=identity, level=ServiceLevel(come_alive_p95_s=bound, warm_p99_ms=250))
            for identity, bound in sorted(bounds.items())
        )
    )


def generation_level(identity, bound) -> IdentityServiceLevel:
    return IdentityServiceLevel(
        identity=identity, level=ServiceLevel(come_alive_p95_s=bound, ttft_p99_ms=1000, itl_p99_ms=80)
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
                    DemandHour(start=START + timedelta(hours=hour), requests=count)
                    for hour, count in sorted(counts.items())
                ),
            )
            for identity, counts in sorted((hourly or {}).items())
        ),
    )


def run(service_levels, receipts=(), boots=(), *, current=None, hourly=None, price_list=None, **overrides):
    parameters = PlannerParameters(
        **({"placement": "test", "max_containers": 4, "fallback_floor": 2, "cooldown_s": 900} | overrides)
    )
    return plan(service_levels, receipts, boots, price_list or prices(), demand(hourly), current, parameters)


def lane(name, cell=L4, *, floor=1, shape=SHAPE) -> PlanLane:
    return PlanLane(
        name=name,
        queue_pool="default",
        bundle="default",
        cell=cell,
        container_shape=shape,
        concurrency=8,
        max_containers=4,
        envelope=Envelope(floor_min=floor, floor_max=floor),
    )


def plan_of(lanes, residents) -> Plan:
    return Plan(
        placement="test",
        lanes=tuple(lanes),
        identities=tuple(
            PlanIdentity(identity=identity, lane=name, warmth=warmth)
            for identity, (name, warmth) in sorted(residents.items())
        ),
    )


def placements(result) -> dict[str, tuple[str, Warmth]]:
    return {entry.identity: (entry.lane, entry.warmth) for entry in result.plan.identities}


def rendered(result) -> tuple[str, str]:
    return render(result.plan.to_dict()), render(result.report.to_dict())


def host_and_guest(guest_footprint=4000, guest_working=1500, *, guest_receipt=True):
    receipts = [receipt(HOST, footprint=9000, working=2000)]
    if guest_receipt:
        receipts.append(receipt(GUEST, footprint=guest_footprint, working=guest_working))
    return receipts


def test_a_lazy_resident_joins_a_lane_formed_in_the_same_run_when_its_memory_fits():
    result = run(levels(**{HOST: 0, GUEST: 60}), host_and_guest(), [load(GUEST)])
    assert placements(result) == {HOST: ("org-host-l4", Warmth.PINNED), GUEST: ("org-host-l4", Warmth.LAZY_RESIDENT)}
    assert result.report.monthly_cost == pytest.approx(MONTH)
    chosen = result.report.identity(GUEST).chosen
    assert (chosen.lane, chosen.warmth, chosen.come_alive_s, chosen.monthly_cost) == (
        "org-host-l4",
        Warmth.LAZY_RESIDENT,
        20.0,
        0,
    )
    (planned,) = result.report.lanes
    assert (planned.memory_mib, planned.memory_limit_mib) == (9000 + 4000 + 2000, pytest.approx(LIMIT))
    assert planned.to_dict()["memoryMiB"] == 15000
    again = run(levels(**{HOST: 0, GUEST: 60}), host_and_guest(), [load(GUEST)], current=result.plan)
    assert again.plan == result.plan
    assert again.report.unchanged


def test_a_lane_that_cannot_hold_the_resident_with_its_working_memory_hosts_none():
    result = run(levels(**{HOST: 0, GUEST: 60}), host_and_guest(guest_footprint=8000), [load(GUEST)])
    assert placements(result) == {HOST: ("org-host-l4", Warmth.PINNED), GUEST: ("org-guest-l4", Warmth.PINNED)}
    assert [request for request in result.report.measurement_requests if request.lane == "org-host-l4"] == []


def test_lazy_residents_add_their_largest_footprint_and_the_largest_working_memory():
    receipts = [
        receipt(HOST, footprint=9000, working=2000),
        receipt("org/g1", footprint=4000, working=1000),
        receipt("org/g2", footprint=3000, working=3500),
    ]
    boots = [load("org/g1"), load("org/g2")]
    result = run(levels(**{HOST: 0, "org/g1": 60, "org/g2": 60}), receipts, boots)
    assert placements(result) == {
        HOST: ("org-host-l4", Warmth.PINNED),
        "org/g1": ("org-host-l4", Warmth.LAZY_RESIDENT),
        "org/g2": ("org-host-l4", Warmth.LAZY_RESIDENT),
    }
    (planned,) = result.report.lanes
    assert planned.memory_mib == 9000 + 4000 + 3500
    receipts[2] = receipt("org/g2", footprint=3000, working=6000)
    result = run(levels(**{HOST: 0, "org/g1": 60, "org/g2": 60}), receipts, boots)
    assert placements(result)["org/g2"] == ("org-g2-l4", Warmth.PINNED)
    assert {lane.name: lane.memory_mib for lane in result.report.lanes}["org-host-l4"] == 15000


def test_a_packing_without_footprints_is_a_measurement_request_with_its_saving_bound():
    receipts = [receipt(HOST, footprint=9000, working=2000), receipt(GUEST)]
    result = run(levels(**{HOST: 0, GUEST: 60}), receipts, [load(GUEST)])
    assert placements(result)[GUEST] == ("org-guest-l4", Warmth.PINNED)
    requests = [request for request in result.report.measurement_requests if request.lane == "org-host-l4"]
    assert [(request.identity, request.warmth, request.measure, request.co_residents) for request in requests] == [
        (GUEST, Warmth.PINNED, (Measurement.CO_RESIDENCY_RECEIPT, Measurement.FOOTPRINT), (HOST,)),
        (GUEST, Warmth.LAZY_RESIDENT, (Measurement.FOOTPRINT,), ()),
    ]
    for request in requests:
        assert (request.reason, request.cost_bound, request.concurrency) == (RequestReason.UNMEASURED, 0, 8)
        assert request.saving_bound == pytest.approx(MONTH)
    assert requests[1].to_dict()["measure"] == ["footprint"]


def test_a_packing_without_a_load_asks_for_the_load_and_the_footprints_it_lacks():
    receipts = [receipt(HOST), receipt(GUEST, footprint=4000, working=1500)]
    result = run(levels(**{HOST: 0, GUEST: 60}), receipts)
    lazy = [
        request
        for request in result.report.measurement_requests
        if request.lane == "org-host-l4" and request.warmth is Warmth.LAZY_RESIDENT
    ]
    assert [request.measure for request in lazy] == [(Measurement.BOOT_EVIDENCE, Measurement.FOOTPRINT)]


@pytest.mark.parametrize(
    ("price_list", "shape"), [(prices(cards=False), SHAPE), (prices(), PAIR)], ids=["no-card-memory", "two-cards"]
)
def test_a_lane_whose_card_memory_is_unknown_or_whose_container_holds_several_cards_packs_nothing(price_list, shape):
    receipts = [
        receipt(HOST, footprint=9000, working=2000, shape=shape),
        receipt(GUEST, footprint=4000, working=1500, shape=shape),
    ]
    boots = [replace(load(GUEST), container_shape=shape)]
    result = run(levels(**{HOST: 0, GUEST: 60}), receipts, boots, price_list=price_list)
    assert placements(result)[GUEST] == ("org-guest-l4", Warmth.PINNED)
    assert [request for request in result.report.measurement_requests if request.lane is not None] == []


def pair_receipts(*, a_p99=180.0, b_p99=200.0):
    return [
        receipt("org/a", footprint=6000, working=1000),
        receipt("org/b", footprint=5000, working=1000),
        receipt("org/a", beside=["org/b"], footprint=6000, working=1500, p99=a_p99, rps=8.0),
        receipt("org/b", beside=["org/a"], footprint=5000, working=1500, p99=b_p99, rps=8.0),
    ]


def test_two_never_cold_identities_share_a_card_on_their_co_residency_receipt():
    result = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts())
    assert placements(result) == {"org/a": ("org-b-l4", Warmth.PINNED), "org/b": ("org-b-l4", Warmth.PINNED)}
    assert result.report.monthly_cost == pytest.approx(MONTH)
    for identity, beside in (("org/a", ("org/b",)), ("org/b", ("org/a",))):
        report = result.report.identity(identity)
        assert (report.lane, report.chosen.co_residents) == ("org-b-l4", beside)
        assert report.chosen.to_dict()["coResidents"] == list(beside)
    assert result.report.identity("org/a").chosen.lane == "org-b-l4"
    (planned,) = result.report.lanes
    assert planned.memory_mib == 6000 + 5000 + 1500
    again = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts(), current=result.plan)
    assert again.plan == result.plan
    assert again.report.unchanged
    assert again.report.identity("org/a").chosen.co_residents == ("org/b",)


def test_a_co_residency_receipt_over_a_warm_bound_keeps_the_identities_apart():
    result = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts(b_p99=300.0))
    assert placements(result) == {"org/a": ("org-a-l4", Warmth.PINNED), "org/b": ("org-b-l4", Warmth.PINNED)}
    assert all(request.lane is None for request in result.report.measurement_requests)


def test_a_co_pinned_resident_is_judged_by_its_co_residency_receipt():
    current = plan_of([lane("shared")], {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)})
    kept = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts(), current=current)
    assert kept.plan == current
    assert kept.report.identity("org/a").chosen.co_residents == ("org/b",)
    split = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts(a_p99=300.0), current=current)
    assert placements(split) == {"org/a": ("org-a-l4", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)}
    assert split.report.identity("org/b").chosen.co_residents == ()
    again = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts(a_p99=300.0), current=split.plan)
    assert again.report.unchanged


def test_a_current_co_pinned_lane_without_its_co_residency_receipt_is_kept_and_asks_for_it():
    current = plan_of([lane("shared")], {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)})
    result = run(levels(**{"org/a": 0, "org/b": 0}), pair_receipts()[:2], current=current)
    assert result.plan == current
    for identity, beside in (("org/a", ("org/b",)), ("org/b", ("org/a",))):
        report = result.report.identity(identity)
        assert (report.status, report.unplaced_reason) == (IdentityStatus.KEPT, UnplacedReason.CURRENT_UNMEASURED)
        (request,) = [request for request in result.report.measurement_requests if request.identity == identity]
        assert (request.measure, request.lane, request.co_residents, request.saving_bound) == (
            (Measurement.CO_RESIDENCY_RECEIPT,),
            "shared",
            beside,
            None,
        )


def test_a_co_residency_receipt_beside_more_residents_stands_for_fewer():
    current = plan_of(
        [lane("shared")],
        {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED), "org/c": ("shared", Warmth.PINNED)},
    )
    receipts = [
        receipt("org/a", beside=["org/b", "org/c"], p99=150.0),
        receipt("org/b", beside=["org/a", "org/c"], p99=150.0),
        receipt("org/c", beside=["org/a", "org/b"], p99=300.0),
        receipt("org/c"),
    ]
    service_levels = levels(**{"org/a": 0, "org/b": 0, "org/c": 600})
    result = run(service_levels, receipts, [cold("org/c")], current=current)
    assert placements(result) == {
        "org/a": ("shared", Warmth.PINNED),
        "org/b": ("shared", Warmth.PINNED),
        "org/c": ("org-c-l4", Warmth.COLD_WITH_BOUND),
    }
    assert result.report.identity("org/a").status is IdentityStatus.PLACED
    assert result.report.identity("org/a").chosen.co_residents == ("org/b", "org/c")
    again = run(service_levels, receipts, [cold("org/c")], current=result.plan)
    assert again.report.unchanged
    assert again.report.monthly_cost == pytest.approx(result.report.monthly_cost)


def test_generation_identities_are_never_packed():
    service_levels = ServiceLevels(
        identities=(
            generation_level("org/llm", 0),
            IdentityServiceLevel(identity=GUEST, level=ServiceLevel(come_alive_p95_s=60, warm_p99_ms=250)),
        )
    )
    receipts = [
        generation_receipt("org/llm", footprint=9000, working=2000),
        receipt(GUEST, footprint=2000, working=500),
    ]
    result = run(service_levels, receipts, [load(GUEST)])
    assert placements(result) == {"org/llm": ("org-llm-l4", Warmth.PINNED), GUEST: ("org-guest-l4", Warmth.PINNED)}
    assert [request for request in result.report.measurement_requests if request.lane is not None] == []
    service_levels = ServiceLevels(
        identities=(
            generation_level("org/llm", 600),
            IdentityServiceLevel(identity=HOST, level=ServiceLevel(come_alive_p95_s=0, warm_p99_ms=250)),
        )
    )
    receipts = [generation_receipt("org/llm", footprint=2000, working=500), receipt(HOST, footprint=9000, working=2000)]
    result = run(service_levels, receipts, [load("org/llm")])
    assert placements(result)["org/llm"] == ("org-llm-l4", Warmth.PINNED)


def test_a_generation_identity_on_a_current_lane_hosts_no_lazy_resident():
    current = plan_of([lane("own"), lane("llm")], {"org/llm": ("llm", Warmth.PINNED), GUEST: ("own", Warmth.PINNED)})
    service_levels = ServiceLevels(
        identities=(
            generation_level("org/llm", 0),
            IdentityServiceLevel(identity=GUEST, level=ServiceLevel(come_alive_p95_s=60, warm_p99_ms=250)),
        )
    )
    receipts = [
        generation_receipt("org/llm", footprint=9000, working=2000),
        receipt(GUEST, footprint=2000, working=500),
    ]
    result = run(service_levels, receipts, [load(GUEST)], current=current)
    assert placements(result)[GUEST] == ("own", Warmth.PINNED)
    assert all(request.lane != "llm" for request in result.report.measurement_requests)


def test_packing_fills_the_lane_with_the_most_footprint_in_use_first():
    receipts = [
        receipt("org/h1", footprint=9000, working=1000),
        receipt("org/h2", footprint=4000, working=1000),
        receipt(GUEST, footprint=2000, working=1000),
    ]
    result = run(levels(**{"org/h1": 0, "org/h2": 0, GUEST: 60}), receipts, [load(GUEST)])
    assert placements(result) == {
        "org/h1": ("org-h1-l4", Warmth.PINNED),
        "org/h2": ("org-h2-l4", Warmth.PINNED),
        GUEST: ("org-h1-l4", Warmth.LAZY_RESIDENT),
    }
    requests = [(request.identity, request.lane, request.warmth) for request in result.report.measurement_requests]
    assert ("org/h2", "org-h1-l4", Warmth.PINNED) in requests


def test_packing_tries_lanes_a_never_cold_identity_pins_before_the_others():
    receipts = [
        receipt("org/a-warm", footprint=9000, working=1000),
        receipt("org/b-never-cold", footprint=4000, working=1000),
        receipt(GUEST, footprint=2000, working=1000),
    ]
    service_levels = levels(**{"org/a-warm": 60, "org/b-never-cold": 0, GUEST: 60})
    result = run(service_levels, receipts, [load(GUEST)])
    assert placements(result)[GUEST] == ("org-b-never-cold-l4", Warmth.LAZY_RESIDENT)


def test_a_host_that_can_itself_ride_a_never_cold_lane_takes_its_guests_along_in_one_run():
    receipts = [
        receipt("org/m1", footprint=5000, working=2500, rps=1.0),
        receipt("org/m2", footprint=5000, working=500),
        receipt("org/m3", footprint=9000, working=2500, rps=1.0),
    ]
    boots = [load("org/m1", seconds=45.0), load("org/m2", seconds=5.0), load("org/m3", seconds=45.0)]
    service_levels = levels(**{"org/m1": 600, "org/m2": 0, "org/m3": 60})
    result = run(service_levels, receipts, boots)
    assert placements(result) == {
        "org/m1": ("org-m2-l4", Warmth.LAZY_RESIDENT),
        "org/m2": ("org-m2-l4", Warmth.PINNED),
        "org/m3": ("org-m2-l4", Warmth.LAZY_RESIDENT),
    }
    assert result.report.monthly_cost == pytest.approx(MONTH)
    (planned,) = result.report.lanes
    assert planned.memory_mib == 5000 + 9000 + 2500
    again = run(service_levels, receipts, boots, current=result.plan)
    assert again.report.unchanged


@pytest.mark.parametrize(("footprint", "measure"), [(4000, None), (8000, ()), (None, (Measurement.FOOTPRINT,))])
def test_the_lazy_resident_selection_on_a_current_lane_requires_the_memory_check(footprint, measure):
    current = plan_of([lane("own"), lane("shared")], {HOST: ("shared", Warmth.PINNED), GUEST: ("own", Warmth.PINNED)})
    receipts = host_and_guest(guest_footprint=footprint)
    result = run(levels(**{HOST: 0, GUEST: 60}), receipts, [load(GUEST)], current=current)
    requests = [
        request.measure
        for request in result.report.measurement_requests
        if request.lane == "shared" and request.warmth is Warmth.LAZY_RESIDENT
    ]
    if measure is None:
        assert placements(result)[GUEST] == ("shared", Warmth.LAZY_RESIDENT)
        assert requests == []
    else:
        assert placements(result)[GUEST] == ("own", Warmth.PINNED)
        assert requests == ([measure] if measure else [])


def test_an_identity_whose_neighbours_departure_strips_its_co_residency_evidence_is_decided_again_in_the_run():
    current = plan_of([lane("shared")], {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)})
    receipts = [
        receipt("org/a", beside=["org/b"], p99=100.0),
        receipt("org/b", beside=["org/a"], p99=300.0),
        receipt("org/a", p99=300.0),
        receipt("org/a", cell=A100),
        receipt("org/b", cell=A100),
    ]
    service_levels = levels(**{"org/a": 0, "org/b": 0})
    result = run(service_levels, receipts, current=current)
    assert placements(result) == {
        "org/a": ("org-a-a100-80gb", Warmth.PINNED),
        "org/b": ("org-b-a100-80gb", Warmth.PINNED),
    }
    again = run(service_levels, receipts, current=result.plan)
    assert again.report.unchanged
    assert again.report.monthly_cost == pytest.approx(result.report.monthly_cost)


def test_packing_visits_identities_in_decreasing_floor_cost():
    current = plan_of(
        [lane("host"), lane("double", floor=2)],
        {HOST: ("host", Warmth.PINNED), "org/z": ("double", Warmth.PINNED)},
    )
    receipts = [
        receipt(HOST, footprint=9000, working=1000),
        receipt("org/y", footprint=5000, working=1000),
        receipt("org/z", footprint=5000, working=1000),
        receipt(HOST, beside=["org/y"], footprint=9000, working=1000),
        receipt("org/y", beside=[HOST], footprint=5000, working=1000),
        receipt(HOST, beside=["org/z"], footprint=9000, working=1000),
        receipt("org/z", beside=[HOST], footprint=5000, working=1000),
    ]
    result = run(levels(**{HOST: 0, "org/y": 0, "org/z": 0}), receipts, current=current)
    assert placements(result) == {
        HOST: ("host", Warmth.PINNED),
        "org/y": ("org-y-l4", Warmth.PINNED),
        "org/z": ("host", Warmth.PINNED),
    }
    assert result.report.monthly_cost == pytest.approx(2 * MONTH)
    assert all(request.identity != "org/y" or request.lane is None for request in result.report.measurement_requests)
    again = run(levels(**{HOST: 0, "org/y": 0, "org/z": 0}), receipts, current=result.plan)
    assert again.report.unchanged


def test_a_pinned_identity_stays_beside_a_neighbour_the_lane_stays_for():
    current = plan_of([lane("shared")], {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)})
    receipts = [*pair_receipts(), receipt("org/b", cell=A100)]
    result = run(levels(**{"org/a": 60, "org/b": 0}), receipts, [cold("org/a")], current=current)
    assert result.plan == current
    assert result.report.monthly_cost == pytest.approx(MONTH)
    report = result.report.identity("org/a")
    assert report.chosen.warmth is Warmth.PINNED
    both = run(levels(**{"org/a": 60, "org/b": 60}), receipts, [cold("org/a"), cold("org/b")], current=current)
    assert placements(both) == {
        "org/a": ("org-a-l4", Warmth.COLD_WITH_BOUND),
        "org/b": ("org-b-l4", Warmth.COLD_WITH_BOUND),
    }
    assert both.report.monthly_cost == 0


def test_identities_that_leave_a_lane_together_return_when_sharing_it_costs_less():
    current = plan_of(
        [lane("shared", cell=A100)], {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)}
    )
    receipts = [
        receipt("org/a", cell=A100, beside=["org/b"]),
        receipt("org/b", cell=A100, beside=["org/a"]),
        receipt("org/a", cell=L40S),
        receipt("org/b", cell=L40S),
    ]
    result = run(levels(**{"org/a": 0, "org/b": 0}), receipts, current=current)
    assert result.plan == current
    assert result.report.monthly_cost == pytest.approx(3 * MONTH)
    alone = plan_of([lane("shared", cell=A100)], {"org/a": ("shared", Warmth.PINNED)})
    single = run(levels(**{"org/a": 0}), receipts, current=alone)
    assert placements(single) == {"org/a": ("org-a-l40s", Warmth.PINNED)}


IDENTITIES = ("org/m0", "org/m1", "org/m2", "org/m3", "org/m4")


@pytest.mark.parametrize("seed", range(150))
def test_random_estates_pack_only_what_fits_and_holds_every_service_level(seed):
    rng = random.Random(seed)  # noqa: S311 -- reproducible test data
    identities = list(IDENTITIES[: rng.randint(2, 5)])
    generation = {identity for identity in identities if rng.random() < 0.15}
    entries = []
    for identity in identities:
        bound = rng.choice([0, 0, 30, 60, 600])
        if identity in generation:
            entries.append(generation_level(identity, bound))
        else:
            entries.append(
                IdentityServiceLevel(
                    identity=identity, level=ServiceLevel(come_alive_p95_s=bound, warm_p99_ms=rng.choice([150, 250]))
                )
            )
    service_levels = ServiceLevels(identities=tuple(entries))
    footprints = {identity: rng.choice([None, 2000, 5000, 9000]) for identity in identities}
    working = {identity: rng.choice([None, 500, 2500]) for identity in identities}
    receipts = []
    for identity in identities:
        for cell in (L4, A100):
            if rng.random() < 0.8:
                if identity in generation:
                    receipts.append(generation_receipt(identity, cell, footprint=footprints[identity], working=500))
                else:
                    receipts.append(
                        receipt(
                            identity,
                            cell,
                            footprint=footprints[identity],
                            working=working[identity],
                            p99=rng.choice([100.0, 200.0]),
                            rps=rng.choice([1.0, 10.0]),
                        )
                    )
    plain = [identity for identity in identities if identity not in generation]
    for _ in range(rng.randint(0, 4)):
        group = rng.sample(plain, min(len(plain), rng.randint(2, 3))) if len(plain) >= 2 else []
        for identity in group:
            receipts.append(
                receipt(
                    identity,
                    beside=sorted(set(group) - {identity}),
                    footprint=footprints[identity],
                    working=rng.choice([500, 2500]),
                    p99=rng.choice([150.0, 300.0]),
                    rps=rng.choice([1.0, 10.0]),
                )
            )
    boots = [load(identity, seconds=rng.choice([5.0, 45.0, 200.0])) for identity in identities if rng.random() < 0.7]
    boots.extend(cold(identity, seconds=rng.choice([20.0, 300.0])) for identity in identities if rng.random() < 0.3)
    hourly = {
        identity: {hour: rng.choice([1, 500]) for hour in rng.sample(range(24), rng.randint(0, 3))}
        for identity in identities
    }
    current = None
    if rng.random() < 0.5:
        first = run(service_levels, receipts, boots, hourly=hourly)
        current = first.plan
    result = run(service_levels, receipts, boots, hourly=hourly, current=current)
    assert Plan.from_dict(result.plan.to_dict()) == result.plan
    levels_by_identity = {entry.identity: entry.level for entry in service_levels.identities}
    residents: dict[str, list[PlanIdentity]] = {}
    for entry in result.plan.identities:
        residents.setdefault(entry.lane, []).append(entry)
    before = {} if current is None else {planned.name: planned for planned in current.lanes}
    reports = {lane_report.name: lane_report for lane_report in result.report.lanes}
    for planned in result.plan.lanes:
        hosted = residents[planned.name]
        if any(entry.identity in generation for entry in hosted):
            assert len(hosted) == 1
        if len(hosted) < 2:
            continue
        pinned = {entry.identity for entry in hosted if entry.warmth is Warmth.PINNED}
        formed = planned.name not in before or {
            entry.identity for entry in current.identities if entry.lane == planned.name
        } != {entry.identity for entry in hosted}
        if formed:
            assert reports[planned.name].memory_mib is not None
            assert reports[planned.name].memory_mib <= reports[planned.name].memory_limit_mib
        for entry in hosted:
            report = result.report.identity(entry.identity)
            if report.chosen is None:
                continue
            level = levels_by_identity[entry.identity]
            if level.warm_p99_ms is not None:
                assert report.chosen.blocked_by == ()
            if entry.warmth is Warmth.PINNED and len(pinned) > 1:
                assert set(report.chosen.co_residents) >= pinned - {entry.identity}
            if entry.warmth is Warmth.LAZY_RESIDENT:
                assert level.come_alive_p95_s != 0
                assert level.come_alive_p95_s is None or report.chosen.come_alive_s <= level.come_alive_p95_s
    for request in result.report.measurement_requests:
        assert request.saving_bound is None or request.saving_bound > 0
    again = run(service_levels, receipts, boots, hourly=hourly, current=result.plan)
    assert again.plan == result.plan
    assert again.report.unchanged
    assert again.report.monthly_cost == pytest.approx(result.report.monthly_cost)
    shuffled = list(receipts)
    rng.shuffle(shuffled)
    reordered = run(service_levels, shuffled, list(reversed(boots)), hourly=hourly, current=current)
    assert rendered(reordered) == rendered(result)
    for entry in service_levels.identities:
        bound = entry.level.come_alive_p95_s
        looser = ServiceLevels(
            identities=tuple(
                replace(other, level=replace(other.level, come_alive_p95_s=bound + 600))
                if other.identity == entry.identity
                else other
                for other in service_levels.identities
            )
        )
        loosened = run(looser, receipts, boots, hourly=hourly, current=result.plan)
        assert loosened.report.monthly_cost <= result.report.monthly_cost + 0.01
