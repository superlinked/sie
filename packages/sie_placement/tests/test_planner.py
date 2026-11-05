from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sie_placement.planner import (
    PlannerInputError,
    PlannerParameters,
    container_hourly_cost,
    envelope_for,
    plan,
)
from sie_placement.report import (
    Blocker,
    ComeAliveBasis,
    IdentityStatus,
    Measurement,
    RefusedRecord,
    RequestReason,
    UnplacedReason,
    render,
)
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
    SchemaError,
    ServiceLevel,
    ServiceLevels,
    SourceRef,
    Throughput,
    Warmth,
)

FIXTURES = Path(__file__).parent / "fixtures" / "planner"
START = datetime(2026, 11, 2, tzinfo=UTC)
WINDOW_HOURS = 24
MONTH = 730.0
SCALE = MONTH / WINDOW_HOURS

L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
L40S = Cell(machine_profile="l40s", gpu_product="NVIDIA-L40S")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
WIDE = ContainerShape(gpu_count=1, cpu_cores=8, memory_mib=32768)
IDENTITY = "org/model"

ESTATE_PARAMETERS = PlannerParameters(
    placement="us-central1",
    max_containers=4,
    fallback_floor=2,
    cooldown_s=900,
    buffer_containers=1,
    bundles={"Qwen/Qwen3-8B": "sglang"},
)


def profile(cell: Cell, accelerator: float, core: float = 0.0) -> MachineProfilePrice:
    return MachineProfilePrice(
        name=cell.machine_profile,
        gpu_product=cell.gpu_product,
        accelerator_per_hour=accelerator,
        core_per_hour=core,
        memory_gib_per_hour=0.0,
    )


def price_list(*, l4=1.0, a100=3.0, core=0.0, multiplier=1.0, extra=()) -> PriceList:
    profiles = [profile(cell, rate, core) for cell, rate in ((L4, l4), (A100, a100)) if rate is not None]
    return PriceList(
        currency="USD",
        machine_profiles=(*profiles, *extra),
        placements=(Placement(name="test", region_multiplier=multiplier),),
    )


L4_ONLY = price_list(a100=None)


def receipt(identity=IDENTITY, cell=L4, *, shape=SHAPE, concurrency=8, rps=10.0, p99=100.0, uri=None) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=shape,
        concurrency=concurrency,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=rps),
        latency=Percentiles(p99_ms=p99),
        source=SourceRef(uri=uri or f"receipts/{identity}/{cell.machine_profile}-{concurrency}-{shape.memory_mib}"),
    )


def generation_receipt(identity=IDENTITY, cell=A100, *, ttft=1000.0, itl=40.0, rps=2.0) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        concurrency=16,
        request_shape=RequestShape(operation=Operation.GENERATE, items_per_request=1, output_tokens=256),
        throughput=Throughput(requests_per_second=rps, output_tokens_per_second=rps * 256),
        ttft=Percentiles(p99_ms=ttft),
        itl=Percentiles(p99_ms=itl),
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}-generate"),
    )


def boot(
    identity=IDENTITY,
    cell=L4,
    warmth=Warmth.COLD_WITH_BOUND,
    seconds=60.0,
    *,
    outcome=BootOutcome.SUCCEEDED,
    borrowed=False,
    shape=SHAPE,
    uri="boots/example.json",
) -> BootEvidence:
    return BootEvidence(
        identity=identity,
        cell=cell,
        container_shape=shape,
        warmth=warmth,
        boot_seconds=seconds,
        outcome=outcome,
        borrowed=borrowed,
        source=SourceRef(uri=uri),
    )


def levels(*entries: tuple[str, ServiceLevel]) -> ServiceLevels:
    return ServiceLevels(identities=tuple(IdentityServiceLevel(identity=name, level=lvl) for name, lvl in entries))


def level(**fields) -> ServiceLevels:
    return levels((IDENTITY, ServiceLevel(**fields)))


def demand(hourly=None) -> Demand:
    return Demand(
        start=START,
        end=START + timedelta(hours=WINDOW_HOURS),
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


def run(service_levels, receipts=(), boots=(), *, prices=None, hourly=None, current=None, **overrides):
    return plan(
        service_levels,
        receipts,
        boots,
        prices or price_list(),
        demand(hourly),
        current,
        parameters(**overrides),
    )


def single_lane_plan(
    name="shared",
    warmth=Warmth.PINNED,
    cell=L4,
    identity=IDENTITY,
    *,
    others=(),
    concurrency=8,
    max_containers=2,
    envelope=None,
) -> Plan:
    floor = 1 if warmth is Warmth.PINNED else 0
    return Plan(
        placement="test",
        lanes=(
            PlanLane(
                name=name,
                queue_pool="default",
                bundle="default",
                cell=cell,
                container_shape=SHAPE,
                concurrency=concurrency,
                max_containers=max_containers,
                envelope=envelope or Envelope(floor_min=floor, floor_max=max(floor, 1)),
            ),
        ),
        identities=tuple(
            PlanIdentity(identity=resident, lane=name, warmth=warmth) for resident in sorted((identity, *others))
        ),
    )


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def estate() -> dict:
    return {
        "service_levels": ServiceLevels.from_dict(load("service-levels.json")),
        "receipts": [Receipt.from_dict(document) for document in load("receipts.json")],
        "boot_evidence": [BootEvidence.from_dict(document) for document in load("boot-evidence.json")],
        "price_list": PriceList.from_dict(load("price-list.json")),
        "demand": Demand.from_dict(load("demand.json")),
        "current_plan": Plan.from_dict(load("current-plan.json")),
        "parameters": ESTATE_PARAMETERS,
    }


def rendered(result) -> tuple[str, str]:
    return render(result.plan.to_dict()), render(result.report.to_dict())


def loosened(service_levels: ServiceLevels, identity: str, bound: int) -> ServiceLevels:
    return replace(
        service_levels,
        identities=tuple(
            replace(entry, level=replace(entry.level, come_alive_p95_s=bound)) if entry.identity == identity else entry
            for entry in service_levels.identities
        ),
    )


LEVEL_FIELDS = {
    "comeAliveP95S": "come_alive_p95_s",
    "warmP99Ms": "warm_p99_ms",
    "ttftP99Ms": "ttft_p99_ms",
    "itlP99Ms": "itl_p99_ms",
}


def relaxed(service_levels: ServiceLevels, identity: str, blockers) -> ServiceLevels:
    fields = {LEVEL_FIELDS[blocker.field]: max(1, math.ceil(blocker.measured)) for blocker in blockers}
    return replace(
        service_levels,
        identities=tuple(
            replace(entry, level=replace(entry.level, **fields)) if entry.identity == identity else entry
            for entry in service_levels.identities
        ),
    )


def test_container_hourly_cost_prices_accelerators_cores_and_memory():
    price = MachineProfilePrice(
        name="l4", gpu_product="NVIDIA-L4", accelerator_per_hour=2.0, core_per_hour=0.05, memory_gib_per_hour=0.01
    )
    shape = ContainerShape(gpu_count=2, cpu_cores=8, memory_mib=20480)
    assert container_hourly_cost(price, shape, 1.5) == pytest.approx(1.5 * (2 * 2.0 + 8 * 0.05 + 20 * 0.01))


def test_a_pinned_cell_costs_its_floor_plus_the_hours_demand_needs_above_it():
    result = run(level(come_alive_p95_s=0), [receipt(rps=1.0)], prices=L4_ONLY, hourly={IDENTITY: {3: 9000, 4: 100}})
    report = result.report.identity(IDENTITY)
    assert report.status is IdentityStatus.PLACED
    assert report.chosen.warmth is Warmth.PINNED
    assert report.chosen.come_alive_basis is ComeAliveBasis.WARM_FLOOR
    assert report.chosen.monthly_cost == pytest.approx(MONTH + (3 - 1) * SCALE)
    assert report.peak_containers == 3
    assert report.demand_requests == 9100


def test_demand_sums_its_operations_in_each_hour():
    hour = START + timedelta(hours=3)
    mixed = Demand(
        start=START,
        end=START + timedelta(hours=WINDOW_HOURS),
        series=tuple(
            DemandSeries(identity=IDENTITY, operation=operation, hours=(DemandHour(start=hour, requests=requests),))
            for operation, requests in ((Operation.ENCODE, 5000), (Operation.SCORE, 4000))
        ),
    )
    result = plan(level(come_alive_p95_s=0), [receipt(rps=1.0)], (), L4_ONLY, mixed, None, parameters())
    report = result.report.identity(IDENTITY)
    assert report.demand_requests == 9000
    assert report.peak_containers == 3
    assert report.chosen.monthly_cost == pytest.approx(MONTH + (3 - 1) * SCALE)


def test_needed_containers_stop_at_the_reviewed_ceiling():
    result = run(
        level(come_alive_p95_s=0), [receipt(rps=1.0)], prices=L4_ONLY, hourly={IDENTITY: {0: 36000}}, max_containers=2
    )
    report = result.report.identity(IDENTITY)
    assert report.chosen.monthly_cost == pytest.approx(MONTH + (2 - 1) * SCALE)
    assert report.peak_containers == 10


def test_a_cold_cell_costs_its_work_and_the_tail_each_burst_holds():
    result = run(
        level(come_alive_p95_s=120), [receipt(rps=10.0)], [boot()], prices=L4_ONLY, hourly={IDENTITY: {0: 1, 5: 10}}
    )
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.warmth is Warmth.COLD_WITH_BOUND
    assert chosen.come_alive_s == 60.0
    one_burst = 1 / 10.0 + 1 * 2 * 900
    capped_hour = 2 * 3600
    assert chosen.monthly_cost == pytest.approx((one_burst + capped_hour) / 3600 * SCALE)


def test_a_saturated_cold_hour_runs_at_most_the_ceiling():
    result = run(
        level(come_alive_p95_s=120),
        [receipt(rps=1.0)],
        [boot()],
        prices=L4_ONLY,
        hourly={IDENTITY: {0: 36000}},
        max_containers=2,
    )
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.warmth is Warmth.COLD_WITH_BOUND
    assert chosen.monthly_cost == pytest.approx(2 * SCALE)


def test_an_active_lane_without_a_fallback_floor_holds_one_container():
    result = run(
        level(come_alive_p95_s=120),
        [receipt(rps=10.0)],
        [boot()],
        prices=L4_ONLY,
        hourly={IDENTITY: {0: 2}},
        fallback_floor=0,
        cooldown_s=60,
    )
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.monthly_cost == pytest.approx((2 / 10.0 + 2 * 1 * 60) / 3600 * SCALE)


def test_the_active_floor_never_exceeds_the_ceiling():
    result = run(
        level(come_alive_p95_s=120),
        [receipt(rps=10.0)],
        [boot()],
        prices=L4_ONLY,
        hourly={IDENTITY: {0: 1}},
        fallback_floor=3,
        max_containers=1,
    )
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.monthly_cost == pytest.approx((1 / 10.0 + 1 * 1 * 900) / 3600 * SCALE)


def test_an_idle_cold_cell_costs_nothing_and_its_lane_keeps_no_floor():
    result = run(level(come_alive_p95_s=120), [receipt()], [boot()], prices=L4_ONLY)
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.warmth is Warmth.COLD_WITH_BOUND
    assert chosen.monthly_cost == 0
    (lane,) = result.plan.lanes
    assert lane.envelope == Envelope(floor_min=0, floor_max=0)
    assert result.plan.identities == (PlanIdentity(identity=IDENTITY, lane=lane.name, warmth=Warmth.COLD_WITH_BOUND),)


def test_a_cold_posture_without_boot_evidence_is_a_measurement_request():
    result = run(level(come_alive_p95_s=120), [receipt()], prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    (request,) = result.report.measurement_requests
    assert request.measure == (Measurement.BOOT_EVIDENCE,)
    assert request.warmth is Warmth.COLD_WITH_BOUND
    assert request.reason is RequestReason.UNMEASURED
    assert (request.cell, request.container_shape) == (L4, SHAPE)
    assert request.cost_bound == 0
    assert request.saving_bound == pytest.approx(MONTH)


@pytest.mark.parametrize("reverse", [False, True])
def test_a_cell_without_boot_evidence_asks_once_at_its_cheapest_receipt(reverse):
    receipts = [receipt(rps=0.01), receipt(concurrency=16, rps=0.02)]
    result = run(
        level(come_alive_p95_s=120),
        receipts[::-1] if reverse else receipts,
        prices=L4_ONLY,
        hourly={IDENTITY: {0: 1}},
    )
    (request,) = result.report.measurement_requests
    assert request.measure == (Measurement.BOOT_EVIDENCE,)
    assert request.cost_bound == pytest.approx((1 / 0.02 + 1 * 2 * 900) / 3600 * SCALE)


def test_snapshot_restores_are_never_requested():
    result = run(level(come_alive_p95_s=120), [receipt()], [boot(seconds=300.0)], prices=L4_ONLY)
    report = result.report
    assert report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    assert report.measurement_requests == ()
    assert [lever.warmth for lever in report.come_alive_levers] == [Warmth.COLD_WITH_BOUND]


def test_a_failed_boot_is_not_selectable_and_asks_for_a_new_measurement():
    boots = [boot(seconds=900.0, outcome=BootOutcome.TIMED_OUT)]
    result = run(level(come_alive_p95_s=600), [receipt()], boots, prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    (request,) = result.report.measurement_requests
    assert request.reason is RequestReason.FAILED
    assert result.report.come_alive_levers == ()


def test_a_failed_boot_disqualifies_the_successful_ones_beside_it():
    boots = [boot(seconds=40.0, uri="boots/1.json"), boot(seconds=0.0, outcome=BootOutcome.FAILED, uri="boots/2.json")]
    result = run(level(come_alive_p95_s=600), [receipt()], boots, prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED


def test_the_slowest_successful_boot_stands_for_the_percentile():
    boots = [boot(seconds=80.0, uri="boots/1.json"), boot(seconds=130.0, uri="boots/2.json")]
    result = run(level(come_alive_p95_s=120), [receipt()], boots, prices=L4_ONLY)
    report = result.report.identity(IDENTITY)
    assert report.chosen.warmth is Warmth.PINNED
    assert report.next_cheaper.blocked_by == (Blocker(field="comeAliveP95S", bound=120, measured=130.0),)


def test_boot_seconds_render_the_same_whatever_their_number_type():
    boots = [boot(seconds=60, uri="boots/1.json"), boot(seconds=60.0, uri="boots/2.json")]
    first = run(level(come_alive_p95_s=120), [receipt()], boots, prices=L4_ONLY)
    second = run(level(come_alive_p95_s=120), [receipt()], boots[::-1], prices=L4_ONLY)
    assert rendered(first) == rendered(second)
    assert first.report.to_dict()["identities"][0]["chosen"]["comeAliveS"] == 60.0


def test_own_boot_evidence_replaces_borrowed_evidence():
    borrowed = boot(seconds=150.0, borrowed=True, uri="boots/borrowed.json")
    own = boot(seconds=60.0, uri="boots/own.json")
    result = run(level(come_alive_p95_s=120), [receipt()], [borrowed, own], prices=L4_ONLY)
    chosen = result.report.identity(IDENTITY).chosen
    assert (chosen.warmth, chosen.come_alive_s) == (Warmth.COLD_WITH_BOUND, 60.0)
    assert chosen.come_alive_basis is ComeAliveBasis.BOOT_EVIDENCE
    failed = boot(seconds=0.0, outcome=BootOutcome.FAILED, uri="boots/own-failed.json")
    result = run(level(come_alive_p95_s=200), [receipt()], [boot(seconds=60.0, borrowed=True), failed], prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED


def test_borrowed_boot_evidence_stands_in_and_is_labelled():
    result = run(level(come_alive_p95_s=120), [receipt()], [boot(seconds=60.0, borrowed=True)], prices=L4_ONLY)
    chosen = result.report.identity(IDENTITY).chosen
    assert (chosen.warmth, chosen.come_alive_s) == (Warmth.COLD_WITH_BOUND, 60.0)
    assert chosen.come_alive_basis is ComeAliveBasis.BORROWED_BOOT_EVIDENCE
    assert chosen.to_dict()["comeAliveBasis"] == "borrowed-boot-evidence"


def test_a_zero_come_alive_bound_never_selects_a_cold_posture():
    boots = [boot(seconds=0.0), boot(warmth=Warmth.SNAPSHOT_COLD, seconds=0.0, uri="boots/snapshot.json")]
    result = run(level(come_alive_p95_s=0), [receipt()], boots, prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    assert result.report.measurement_requests == ()


def test_an_undeclared_come_alive_bound_admits_any_measured_boot():
    result = run(level(warm_p99_ms=250), [receipt()], [boot(seconds=5000.0)], prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.COLD_WITH_BOUND


def test_a_snapshot_restore_under_the_bound_beats_a_slow_boot():
    boots = [boot(seconds=300.0), boot(warmth=Warmth.SNAPSHOT_COLD, seconds=20.0, uri="boots/snapshot.json")]
    result = run(level(come_alive_p95_s=60), [receipt()], boots, prices=L4_ONLY)
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.warmth is Warmth.SNAPSHOT_COLD
    assert chosen.come_alive_s == 20.0
    assert result.plan.identities[0].warmth is Warmth.SNAPSHOT_COLD


def test_equal_costs_prefer_the_faster_come_alive():
    boots = [boot(seconds=50.0), boot(warmth=Warmth.SNAPSHOT_COLD, seconds=20.0, uri="boots/snapshot.json")]
    result = run(level(come_alive_p95_s=60), [receipt()], boots, prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.SNAPSHOT_COLD


def test_a_cell_whose_receipt_misses_the_warm_bound_is_never_chosen():
    receipts = [receipt(cell=L4, p99=300.0), receipt(cell=A100, p99=120.0)]
    result = run(level(come_alive_p95_s=0, warm_p99_ms=250), receipts)
    report = result.report.identity(IDENTITY)
    assert report.chosen.cell == A100
    assert report.chosen.monthly_cost == pytest.approx(3 * MONTH)
    assert report.next_cheaper.cell == L4
    assert report.next_cheaper.blocked_by == (Blocker(field="warmP99Ms", bound=250, measured=300.0),)
    assert report.next_cheaper.saving == pytest.approx(2 * MONTH)
    assert result.plan.lanes[0].cell == A100


def test_generation_cells_meet_both_the_first_token_and_the_per_token_bound():
    receipts = [generation_receipt(cell=L4, ttft=2000.0, itl=95.0), generation_receipt(cell=A100, ttft=1200.0)]
    result = run(level(come_alive_p95_s=0, ttft_p99_ms=1500, itl_p99_ms=80), receipts)
    report = result.report.identity(IDENTITY)
    assert report.chosen.cell == A100
    assert report.next_cheaper.blocked_by == (
        Blocker(field="ttftP99Ms", bound=1500, measured=2000.0),
        Blocker(field="itlP99Ms", bound=80, measured=95.0),
    )
    assert report.next_cheaper.saving == pytest.approx(2 * MONTH)


def test_a_receipt_whose_realized_product_differs_from_its_profile_is_refused():
    faster = Cell(machine_profile="l4", gpu_product="NVIDIA-L40S")
    result = run(level(come_alive_p95_s=0), [receipt(cell=faster), receipt(cell=A100)])
    report = result.report.identity(IDENTITY)
    assert report.chosen.cell == A100
    (refusal,) = result.report.refusals
    assert refusal.record is RefusedRecord.RECEIPT
    assert refusal.cell == faster
    assert refusal.reason == (
        "realized GPU product 'NVIDIA-L40S' differs from 'NVIDIA-L4', the product machine profile 'l4' requests"
    )
    (request,) = result.report.measurement_requests
    assert (request.cell, request.reason, request.measure) == (L4, RequestReason.REFUSED, (Measurement.RECEIPT,))
    assert request.saving_bound == pytest.approx(2 * MONTH)
    assert all(lane.cell != faster for lane in result.plan.lanes)


def test_boot_evidence_whose_realized_product_differs_from_its_profile_is_refused():
    other = Cell(machine_profile="l4", gpu_product="NVIDIA-A10G")
    result = run(level(come_alive_p95_s=120), [receipt()], [boot(cell=other)], prices=L4_ONLY)
    assert result.report.identity(IDENTITY).chosen.warmth is Warmth.PINNED
    (refusal,) = result.report.refusals
    assert refusal.record is RefusedRecord.BOOT_EVIDENCE
    (request,) = result.report.measurement_requests
    assert (request.measure, request.reason) == ((Measurement.BOOT_EVIDENCE,), RequestReason.UNMEASURED)


def test_a_receipt_on_an_unpriced_profile_is_refused():
    unpriced = Cell(machine_profile="t4", gpu_product="NVIDIA-T4")
    result = run(level(come_alive_p95_s=0), [receipt(cell=unpriced)], prices=L4_ONLY)
    report = result.report.identity(IDENTITY)
    assert report.status is IdentityStatus.UNPLACED
    assert report.unplaced_reason is UnplacedReason.NO_RECEIPT
    assert result.report.refusals[0].reason == "machine profile 't4' has no price"
    (request,) = result.report.measurement_requests
    assert (request.cell, request.reason, request.saving_bound) == (L4, RequestReason.UNMEASURED, None)


@pytest.mark.parametrize(
    ("service_level", "candidate", "reason"),
    [
        (
            ServiceLevel(ttft_p99_ms=1000, itl_p99_ms=50),
            receipt(),
            "the receipt reports one latency and the service level declares ttftP99Ms and itlP99Ms",
        ),
        (
            ServiceLevel(warm_p99_ms=250),
            generation_receipt(cell=L4),
            "the receipt reports ttft and itl and the service level declares warmP99Ms",
        ),
    ],
)
def test_a_receipt_that_cannot_answer_the_service_level_is_refused(service_level, candidate, reason):
    result = run(levels((IDENTITY, service_level)), [candidate], prices=L4_ONLY)
    assert result.report.identity(IDENTITY).status is IdentityStatus.UNPLACED
    assert [refusal.reason for refusal in result.report.refusals] == [reason]


def test_records_of_identities_without_a_service_level_are_ignored():
    stranger = Cell(machine_profile="l4", gpu_product="NVIDIA-A10G")
    result = run(
        level(come_alive_p95_s=0),
        [receipt(), receipt(identity="org/other", cell=stranger)],
        [boot(identity="org/other", cell=stranger)],
        prices=L4_ONLY,
    )
    assert result.report.refusals == ()
    assert [entry.identity for entry in result.plan.identities] == [IDENTITY]
    assert [entry.identity for entry in result.report.identities] == [IDENTITY]


def test_an_unmeasured_cell_is_bounded_by_its_accelerator_price_alone():
    result = run(level(come_alive_p95_s=0), [receipt(cell=A100)], prices=price_list(core=0.5))
    chosen = result.report.identity(IDENTITY).chosen
    assert chosen.monthly_cost == pytest.approx((3.0 + 4 * 0.5) * MONTH)
    (request,) = result.report.measurement_requests
    assert (request.cell, request.measure, request.warmth) == (L4, (Measurement.RECEIPT,), Warmth.PINNED)
    assert request.cost_bound == pytest.approx(1.0 * MONTH)
    assert request.saving_bound == pytest.approx((5.0 - 1.0) * MONTH)


def test_an_unmeasured_cell_under_a_finite_bound_is_bounded_by_its_burst_tails():
    result = run(level(come_alive_p95_s=300), [receipt(cell=A100)], hourly={IDENTITY: {0: 1}})
    requests = {request.cell: request for request in result.report.measurement_requests}
    request = requests[L4]
    assert request.measure == (Measurement.RECEIPT, Measurement.BOOT_EVIDENCE)
    assert request.warmth is Warmth.COLD_WITH_BOUND
    assert request.cost_bound == pytest.approx(1.0 * (1 * 2 * 900) / 3600 * SCALE)
    assert requests[A100].measure == (Measurement.BOOT_EVIDENCE,)


def test_a_cell_that_cannot_save_is_not_requested():
    result = run(level(come_alive_p95_s=0), [receipt(cell=L4)])
    assert result.report.measurement_requests == ()


def test_measurement_requests_rank_unplaced_identities_first_then_by_saving():
    service_levels = levels(
        ("org/big", ServiceLevel(come_alive_p95_s=0)),
        ("org/small", ServiceLevel(come_alive_p95_s=120)),
        ("org/unplaced", ServiceLevel(come_alive_p95_s=0)),
    )
    receipts = [receipt(identity="org/big", cell=A100), receipt(identity="org/small", cell=L4)]
    result = run(service_levels, receipts)
    ranked = [
        (request.identity, request.cell.machine_profile, request.measure, request.saving_bound)
        for request in result.report.measurement_requests
    ]
    assert ranked == [
        ("org/unplaced", "a100-80gb", (Measurement.RECEIPT,), None),
        ("org/unplaced", "l4", (Measurement.RECEIPT,), None),
        ("org/big", "l4", (Measurement.RECEIPT,), pytest.approx(2 * MONTH)),
        ("org/small", "a100-80gb", (Measurement.RECEIPT, Measurement.BOOT_EVIDENCE), pytest.approx(MONTH)),
        ("org/small", "l4", (Measurement.BOOT_EVIDENCE,), pytest.approx(MONTH)),
    ]


def test_the_next_cheaper_option_is_the_nearest_cheaper_one_and_names_its_bound():
    prices = price_list(extra=(profile(L40S, 2.0),))
    receipts = [receipt(cell=L4, p99=400.0), receipt(cell=L40S, p99=300.0), receipt(cell=A100, p99=100.0)]
    result = run(level(come_alive_p95_s=0, warm_p99_ms=250), receipts, prices=prices)
    report = result.report.identity(IDENTITY)
    assert report.chosen.cell == A100
    assert report.next_cheaper.cell == L40S
    assert report.next_cheaper.saving == pytest.approx(MONTH)
    assert report.next_cheaper.blocked_by == (Blocker(field="warmP99Ms", bound=250, measured=300.0),)


@pytest.mark.parametrize(("others", "saving"), [((), MONTH), (("org/kept",), 0.0)])
def test_a_next_cheaper_saving_is_what_loosening_its_bounds_saves_the_plan(others, saving):
    service_levels = levels(*((name, ServiceLevel(come_alive_p95_s=0)) for name in (IDENTITY, *others)))
    current = single_lane_plan(others=others)
    result = run(service_levels, [receipt()], [boot(seconds=30.0)], prices=L4_ONLY, current=current)
    report = result.report.identity(IDENTITY)
    assert (report.lane, report.chosen.warmth) == ("shared", Warmth.PINNED)
    assert (report.next_cheaper.warmth, report.next_cheaper.monthly_cost) == (Warmth.COLD_WITH_BOUND, 0.0)
    assert report.chosen.monthly_cost - report.next_cheaper.monthly_cost == pytest.approx(MONTH)
    assert report.next_cheaper.saving == pytest.approx(saving)
    again = run(
        loosened(service_levels, IDENTITY, 30), [receipt()], [boot(seconds=30.0)], prices=L4_ONLY, current=result.plan
    )
    assert again.report.identity(IDENTITY).chosen.warmth is Warmth.COLD_WITH_BOUND
    assert result.report.monthly_cost - again.report.monthly_cost == pytest.approx(saving)


def test_a_placement_with_nothing_cheaper_names_no_next_option():
    result = run(level(come_alive_p95_s=0), [receipt()], prices=L4_ONLY)
    assert result.report.identity(IDENTITY).next_cheaper is None


@pytest.mark.parametrize(
    ("warmth", "buffer", "fallback", "ceiling", "expected"),
    [
        (Warmth.PINNED, 0, 2, 4, (1, 1)),
        (Warmth.PINNED, 2, 2, 4, (1, 3)),
        (Warmth.PINNED, 5, 2, 4, (1, 4)),
        (Warmth.COLD_WITH_BOUND, 0, 2, 4, (0, 0)),
        (Warmth.COLD_WITH_BOUND, 1, 2, 4, (0, 2)),
        (Warmth.COLD_WITH_BOUND, 3, 2, 4, (0, 3)),
        (Warmth.COLD_WITH_BOUND, 1, 0, 4, (0, 1)),
        (Warmth.COLD_WITH_BOUND, 1, 3, 2, (0, 2)),
        (Warmth.SNAPSHOT_COLD, 1, 2, 4, (0, 2)),
    ],
)
def test_envelopes_add_the_declared_buffer_at_realizable_floors(warmth, buffer, fallback, ceiling, expected):
    envelope = envelope_for(
        warmth, parameters(buffer_containers=buffer, fallback_floor=fallback, max_containers=ceiling)
    )
    assert (envelope.floor_min, envelope.floor_max) == expected
    assert (envelope.buffer_min, envelope.scaledown_window_min_s) == (None, None)


def test_the_plan_carries_the_envelope_and_the_report_prices_its_maximum():
    result = run(level(come_alive_p95_s=0), [receipt()], prices=L4_ONLY, buffer_containers=2)
    (lane,) = result.plan.lanes
    assert lane.envelope == Envelope(floor_min=1, floor_max=3)
    assert (lane.max_containers, lane.concurrency, lane.container_shape) == (4, 8, SHAPE)
    (lane_report,) = result.report.lanes
    assert lane_report.hourly_cost == pytest.approx(1.0)
    assert lane_report.floor_monthly_cost == pytest.approx(MONTH)
    assert lane_report.envelope_max_monthly_cost == pytest.approx(3 * MONTH)
    assert result.report.envelope_max_monthly_cost == pytest.approx(3 * MONTH)
    assert result.report.monthly_cost == pytest.approx(MONTH)
    assert lane_report.monthly_cost == pytest.approx(MONTH)
    assert result.report.identity(IDENTITY).monthly_cost == pytest.approx(MONTH)
    assert result.report.unpriced_lanes == ()


def test_every_declared_queue_wait_bound_is_reported_unattained():
    service_levels = levels(
        ("org/a", ServiceLevel(come_alive_p95_s=0, queue_wait_p99_ms=500)),
        ("org/b", ServiceLevel(come_alive_p95_s=0)),
        ("org/c", ServiceLevel(queue_wait_p99_ms=30000)),
    )
    result = run(service_levels, [receipt(identity="org/a")], prices=L4_ONLY)
    assert result.report.queue_wait_unattained == ("org/a", "org/c")


def test_an_identity_without_a_feasible_cell_keeps_its_current_lane():
    current = single_lane_plan()
    result = run(level(come_alive_p95_s=0, warm_p99_ms=50), [receipt(p99=100.0)], prices=L4_ONLY, current=current)
    assert result.plan == current
    report = result.report.identity(IDENTITY)
    assert report.status is IdentityStatus.KEPT
    assert report.unplaced_reason is UnplacedReason.WARM_BOUND_MISSED
    assert report.lane == "shared"
    assert report.current == current.identities[0]
    assert result.report.unchanged
    assert result.report.lanes[0].floor_monthly_cost == pytest.approx(MONTH)
    assert report.monthly_cost == pytest.approx(MONTH)
    assert report.blocked_by == (Blocker(field="warmP99Ms", bound=50, measured=100.0),)
    assert result.report.monthly_cost == pytest.approx(MONTH)
    assert result.report.to_dict()["totals"] == {
        "monthlyCost": MONTH,
        "envelopeMaxMonthlyCost": MONTH,
        "unpricedLanes": [],
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_a_kept_lane_prices_its_demand_at_the_slowest_receipt_for_its_cell(reverse):
    current = single_lane_plan(max_containers=4, envelope=Envelope(floor_min=1, floor_max=2))
    receipts = [receipt(rps=1.0, p99=100.0, uri="receipts/slow"), receipt(rps=2.0, p99=100.0, uri="receipts/fast")]
    result = run(
        level(come_alive_p95_s=0, warm_p99_ms=50),
        receipts[::-1] if reverse else receipts,
        prices=L4_ONLY,
        hourly={IDENTITY: {3: 9000}},
        current=current,
    )
    assert result.plan == current
    report = result.report.identity(IDENTITY)
    assert report.status is IdentityStatus.KEPT
    assert report.monthly_cost == pytest.approx(MONTH + (3 - 1) * SCALE)
    (lane,) = result.report.lanes
    assert lane.monthly_cost == pytest.approx(MONTH + (3 - 1) * SCALE)
    assert result.report.monthly_cost == pytest.approx(MONTH + (3 - 1) * SCALE)
    assert result.report.envelope_max_monthly_cost == pytest.approx(2 * MONTH)


@pytest.mark.parametrize(
    ("warmth", "ceiling", "hourly", "hours"),
    [
        (Warmth.PINNED, 2, {3: 9000}, MONTH + (2 - 1) * SCALE),
        (Warmth.COLD_WITH_BOUND, 1, {0: 1}, (1 / 1.0 + 1 * 1 * 900) / 3600 * SCALE),
    ],
)
def test_a_kept_lane_prices_its_demand_under_its_own_ceiling(warmth, ceiling, hourly, hours):
    current = single_lane_plan(warmth=warmth, max_containers=ceiling)
    result = run(
        level(come_alive_p95_s=600, warm_p99_ms=50),
        [receipt(rps=1.0, p99=100.0)],
        prices=L4_ONLY,
        hourly={IDENTITY: hourly},
        current=current,
    )
    assert result.plan == current
    assert result.report.identity(IDENTITY).monthly_cost == pytest.approx(hours)


def test_a_kept_lane_without_a_receipt_for_its_demand_lists_it_unpriced():
    current = single_lane_plan()
    result = run(level(come_alive_p95_s=0), [], prices=L4_ONLY, hourly={IDENTITY: {0: 10}}, current=current)
    report = result.report.identity(IDENTITY)
    assert (report.status, report.monthly_cost) == (IdentityStatus.KEPT, None)
    (lane,) = result.report.lanes
    assert lane.unpriced_identities == (IDENTITY,)
    assert lane.monthly_cost == pytest.approx(MONTH)
    assert result.report.unpriced_lanes == ("shared",)
    assert result.report.monthly_cost == pytest.approx(MONTH)


def test_a_kept_lane_on_a_refused_cell_is_kept_and_reported_unpriced():
    other = Cell(machine_profile="l4", gpu_product="NVIDIA-A10G")
    current = single_lane_plan(cell=other)
    result = run(level(come_alive_p95_s=0), [], prices=L4_ONLY, current=current)
    assert result.plan == current
    (lane,) = result.report.lanes
    assert lane.unpriced_reason == (
        "realized GPU product 'NVIDIA-A10G' differs from 'NVIDIA-L4', the product machine profile 'l4' requests"
    )
    assert (lane.hourly_cost, lane.monthly_cost, lane.envelope_max_monthly_cost) == (None, None, None)
    assert result.report.identity(IDENTITY).monthly_cost is None
    assert result.report.to_dict()["totals"] == {
        "monthlyCost": 0.0,
        "envelopeMaxMonthlyCost": 0.0,
        "unpricedLanes": ["shared"],
    }


def test_a_kept_lane_never_exceeds_the_reviewed_ceiling():
    current = single_lane_plan(max_containers=4, envelope=Envelope(floor_min=2, floor_max=3))
    result = run(level(come_alive_p95_s=0), [], prices=L4_ONLY, current=current, max_containers=1)
    (lane,) = result.plan.lanes
    assert (lane.max_containers, lane.envelope) == (1, Envelope(floor_min=1, floor_max=1))
    (change,) = result.report.lane_changes
    assert (change.name, change.before, change.after) == ("shared", current.lanes[0], lane)


@pytest.mark.parametrize(
    ("bound", "boots", "expected"),
    [
        (0, (), (Blocker(field="comeAliveP95S", bound=0, measured=None),)),
        (120, (), ()),
        (120, (boot(seconds=200.0),), (Blocker(field="comeAliveP95S", bound=120, measured=200.0),)),
        (300, (boot(seconds=200.0),), ()),
    ],
)
def test_a_kept_cold_placement_names_the_come_alive_bound_it_is_known_to_miss(bound, boots, expected):
    current = single_lane_plan(warmth=Warmth.COLD_WITH_BOUND)
    result = run(level(come_alive_p95_s=bound), [], boots, prices=L4_ONLY, current=current)
    report = result.report.identity(IDENTITY)
    assert report.status is IdentityStatus.KEPT
    assert report.blocked_by == expected
    assert report.monthly_cost == 0
    if expected and expected[0].measured is None:
        assert report.to_dict()["blockedBy"] == [{"field": "comeAliveP95S", "bound": 0}]


def test_an_identity_without_a_feasible_cell_or_a_current_lane_is_left_out():
    result = run(level(come_alive_p95_s=0), [], prices=L4_ONLY)
    assert (result.plan.lanes, result.plan.identities) == ((), ())
    report = result.report.identity(IDENTITY)
    assert (report.status, report.unplaced_reason) == (IdentityStatus.UNPLACED, UnplacedReason.NO_RECEIPT)
    assert result.report.monthly_cost == 0


def test_lane_names_derive_from_the_identity_and_its_machine_profile():
    result = run(levels(("BAAI/bge-m3:sparse", ServiceLevel())), [receipt(identity="BAAI/bge-m3:sparse")])
    assert result.plan.lanes[0].name == "baai-bge-m3-sparse-l4"


def test_colliding_lane_names_take_an_identity_digest():
    service_levels = levels(("org/model", ServiceLevel()), ("org-model", ServiceLevel()))
    receipts = [receipt(identity="org/model"), receipt(identity="org-model")]
    result = run(service_levels, receipts, prices=L4_ONLY)
    names = {entry.identity: entry.lane for entry in result.plan.identities}
    assert names == {
        identity: f"org-model-l4-{hashlib.sha256(identity.encode()).hexdigest()[:8]}" for identity in names
    }
    assert len(set(names.values())) == 2


def test_a_derived_lane_name_never_takes_a_kept_lane_name():
    current = single_lane_plan(name="org-model-l4", identity="org/kept")
    service_levels = levels(("org/kept", ServiceLevel(come_alive_p95_s=0)), (IDENTITY, ServiceLevel()))
    result = run(service_levels, [receipt()], prices=L4_ONLY, current=current)
    names = {entry.identity: entry.lane for entry in result.plan.identities}
    assert names["org/kept"] == "org-model-l4"
    assert names[IDENTITY] == f"org-model-l4-{hashlib.sha256(IDENTITY.encode()).hexdigest()[:8]}"


def test_a_current_placement_keeps_its_lane_name():
    current = single_lane_plan(name="legacy")
    result = run(level(come_alive_p95_s=0), [receipt()], prices=L4_ONLY, current=current, max_containers=2)
    assert result.plan == current
    assert result.report.identity(IDENTITY).lane == "legacy"
    assert result.report.unchanged


def test_a_current_placement_that_differs_only_in_its_envelope_keeps_its_lane_name():
    current = single_lane_plan(name="legacy", max_containers=8, envelope=Envelope(floor_min=2, floor_max=5))
    result = run(level(come_alive_p95_s=0), [receipt()], prices=L4_ONLY, current=current)
    (lane,) = result.plan.lanes
    assert (lane.name, lane.max_containers, lane.envelope) == ("legacy", 4, Envelope(floor_min=1, floor_max=1))
    assert result.report.identity_changes == ()
    assert [change.name for change in result.report.lane_changes] == ["legacy"]


def test_a_current_lane_that_differs_in_concurrency_is_not_the_current_placement():
    current = single_lane_plan(name="legacy", max_containers=4)
    receipts = [receipt(concurrency=16), receipt(concurrency=8)]
    result = run(level(come_alive_p95_s=0), receipts, prices=L4_ONLY, current=current)
    (lane,) = result.plan.lanes
    assert (lane.name, lane.concurrency) == ("legacy", 8)
    assert result.report.unchanged
    moved = run(level(come_alive_p95_s=0), [receipts[0]], prices=L4_ONLY, current=current)
    (lane,) = moved.plan.lanes
    assert (lane.name, lane.concurrency) == ("org-model-l4", 16)
    assert [(change.name, change.before is None) for change in moved.report.lane_changes] == [
        ("legacy", False),
        ("org-model-l4", True),
    ]


def test_a_current_placement_on_a_shared_lane_stays_beside_its_kept_neighbour():
    current = single_lane_plan(others=("org/kept",), max_containers=4)
    service_levels = levels(
        (IDENTITY, ServiceLevel(come_alive_p95_s=0)), ("org/kept", ServiceLevel(come_alive_p95_s=0))
    )
    result = run(service_levels, [receipt()], prices=L4_ONLY, current=current, hourly={IDENTITY: {3: 300000}})
    assert result.plan == current
    assert result.report.unchanged
    (lane,) = result.report.lanes
    assert lane.identities == ("org/kept", IDENTITY)
    above = (min(4, math.ceil(300000 / 36000)) - 1) * SCALE
    assert lane.monthly_cost == pytest.approx(MONTH + above)
    assert result.report.identity(IDENTITY).monthly_cost == pytest.approx(MONTH + above)
    assert result.report.identity("org/kept").monthly_cost == pytest.approx(MONTH)
    assert result.report.monthly_cost == pytest.approx(MONTH + above)


def test_current_placements_that_share_a_lane_keep_sharing_it():
    current = single_lane_plan(others=("org/other",), max_containers=4)
    service_levels = levels((IDENTITY, ServiceLevel()), ("org/other", ServiceLevel()))
    result = run(service_levels, [receipt(), receipt(identity="org/other")], prices=L4_ONLY, current=current)
    assert result.plan == current
    assert result.report.unchanged
    assert result.report.monthly_cost == pytest.approx(MONTH)
    assert result.report.shared_chart_lanes == ()


def test_a_placement_that_moves_leaves_its_shared_lane_to_the_kept_neighbour():
    current = single_lane_plan(others=("org/kept",), max_containers=4)
    service_levels = levels(
        (IDENTITY, ServiceLevel(come_alive_p95_s=0, warm_p99_ms=250)), ("org/kept", ServiceLevel(come_alive_p95_s=0))
    )
    receipts = [receipt(p99=300.0), receipt(cell=A100)]
    result = run(service_levels, receipts, current=current)
    names = {entry.identity: entry.lane for entry in result.plan.identities}
    assert names == {IDENTITY: "org-model-a100-80gb", "org/kept": "shared"}
    assert result.report.identity(IDENTITY).current == current.identities[1]


@pytest.mark.parametrize(
    ("overrides", "moved"),
    [({"bundles": {IDENTITY: "sglang"}}, ("default", "sglang")), ({"queue_pool": "batch"}, ("batch", "default"))],
)
def test_a_new_queue_pool_or_bundle_moves_the_identity_off_its_current_lane(overrides, moved):
    current = single_lane_plan(others=("org/kept",), max_containers=4)
    service_levels = levels(
        (IDENTITY, ServiceLevel(come_alive_p95_s=0)), ("org/kept", ServiceLevel(come_alive_p95_s=0))
    )
    result = run(service_levels, [receipt()], prices=L4_ONLY, current=current, **overrides)
    lanes = {lane.name: (lane.queue_pool, lane.bundle) for lane in result.plan.lanes}
    assert lanes == {"org-model-l4": moved, "shared": ("default", "default")}


def test_removing_one_of_two_colliding_identities_keeps_the_other_lane_name():
    service_levels = levels(("org/model", ServiceLevel()), ("org-model", ServiceLevel()))
    receipts = [receipt(identity="org/model"), receipt(identity="org-model")]
    first = run(service_levels, receipts, prices=L4_ONLY)
    second = run(levels(("org/model", ServiceLevel())), receipts, prices=L4_ONLY, current=first.plan)
    digest = hashlib.sha256(b"org-model").hexdigest()[:8]
    kept_name = f"org-model-l4-{hashlib.sha256(b'org/model').hexdigest()[:8]}"
    assert [entry.lane for entry in second.plan.identities] == [kept_name]
    assert [(change.identity, change.after) for change in second.report.identity_changes] == [("org-model", None)]
    assert [(change.name, change.after) for change in second.report.lane_changes] == [(f"org-model-l4-{digest}", None)]


def test_plan_lanes_that_name_one_chart_lane_are_reported():
    service_levels = levels(("org/a", ServiceLevel()), ("org/b", ServiceLevel()), ("org/c", ServiceLevel()))
    receipts = [receipt(identity=identity) for identity in ("org/a", "org/b", "org/c")]
    result = run(service_levels, receipts, prices=L4_ONLY, bundles={"org/c": "sglang"})
    (shared,) = result.report.shared_chart_lanes
    assert shared.to_dict() == {
        "queuePool": "default",
        "machineProfile": "l4",
        "bundle": "default",
        "lanes": ["org-a-l4", "org-b-l4"],
    }
    apart = run(service_levels, receipts, prices=L4_ONLY, bundles={"org/a": "a", "org/b": "b"})
    assert apart.report.shared_chart_lanes == ()


def test_a_derived_lane_name_never_takes_a_kept_lane_named_with_its_digest():
    digest = hashlib.sha256(IDENTITY.encode()).hexdigest()
    first = single_lane_plan(name="org-model-l4", identity="org/kept")
    second = single_lane_plan(name=f"org-model-l4-{digest[:8]}", identity="org/kept-too")
    current = Plan(
        placement="test", lanes=(*first.lanes, *second.lanes), identities=(*first.identities, *second.identities)
    )
    service_levels = levels(
        (IDENTITY, ServiceLevel()),
        ("org/kept", ServiceLevel(come_alive_p95_s=0)),
        ("org/kept-too", ServiceLevel(come_alive_p95_s=0)),
    )
    result = run(service_levels, [receipt()], prices=L4_ONLY, current=current)
    names = {entry.identity: entry.lane for entry in result.plan.identities}
    assert names == {
        IDENTITY: f"org-model-l4-{digest}",
        "org/kept": "org-model-l4",
        "org/kept-too": f"org-model-l4-{digest[:8]}",
    }
    assert {lane.name: lane for lane in result.plan.lanes}[f"org-model-l4-{digest[:8]}"] == second.lanes[0]


def test_bundles_and_the_queue_pool_come_from_the_parameters():
    service_levels = levels(("org/embed", ServiceLevel()), ("org/generate", ServiceLevel()))
    result = run(
        service_levels,
        [receipt(identity="org/embed"), receipt(identity="org/generate")],
        prices=L4_ONLY,
        queue_pool="batch",
        bundle="default",
        bundles={"org/generate": "sglang"},
    )
    lanes = {lane.name: (lane.queue_pool, lane.bundle) for lane in result.plan.lanes}
    assert lanes == {"org-embed-l4": ("batch", "default"), "org-generate-l4": ("batch", "sglang")}


def test_equal_costs_prefer_the_current_placement():
    twin = Cell(machine_profile="l4-b", gpu_product="NVIDIA-L4")
    prices = price_list(a100=None, extra=(profile(twin, 1.0),))
    receipts = [receipt(cell=L4), receipt(cell=twin)]
    first = run(level(come_alive_p95_s=0), receipts, prices=prices)
    assert first.plan.lanes[0].cell == L4
    current = single_lane_plan(name="org-model-l4-b", cell=twin)
    second = run(level(come_alive_p95_s=0), receipts, prices=prices, current=current)
    assert second.plan.lanes[0].cell == twin
    assert second.report.identity(IDENTITY).current == current.identities[0]


def test_the_region_multiplier_scales_every_price():
    result = run(level(come_alive_p95_s=0), [receipt()], prices=price_list(a100=None, multiplier=1.25))
    assert result.report.identity(IDENTITY).chosen.monthly_cost == pytest.approx(1.25 * MONTH)
    assert result.report.lanes[0].hourly_cost == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"placement": ""}, "placement"),
        ({"max_containers": 0}, "maxContainers"),
        ({"fallback_floor": -1}, "fallbackFloor"),
        ({"cooldown_s": -1}, "cooldownS"),
        ({"buffer_containers": -1}, "bufferContainers"),
        ({"hours_per_month": 0}, "hoursPerMonth"),
        ({"hours_per_month": math.inf}, "hoursPerMonth"),
        ({"queue_pool": "two words"}, "queuePool"),
        ({"bundle": ""}, "bundle"),
        ({"bundles": {"": "sglang"}}, "bundles"),
        ({"bundles": {IDENTITY: ""}}, "bundles['org/model']"),
    ],
)
def test_parameters_are_validated(overrides, path):
    with pytest.raises(SchemaError) as caught:
        parameters(**overrides)
    assert caught.value.path == path


def test_parameters_echo_into_the_report():
    echoed = parameters(bundles={"org/b": "sglang", "org/a": "default"}).to_dict()
    assert echoed == {
        "placement": "test",
        "maxContainers": 4,
        "fallbackFloor": 2,
        "cooldownS": 900,
        "bufferContainers": 0,
        "hoursPerMonth": 730.0,
        "queuePool": "default",
        "bundle": "default",
        "bundles": {"org/a": "default", "org/b": "sglang"},
    }
    assert list(echoed["bundles"]) == ["org/a", "org/b"]


def test_the_placement_must_be_in_the_price_list():
    with pytest.raises(PlannerInputError, match="placement 'elsewhere' is not in the price list"):
        run(level(), [receipt()], placement="elsewhere")


def test_the_current_plan_must_be_for_the_same_placement():
    current = Plan(placement="other", lanes=(), identities=())
    with pytest.raises(PlannerInputError, match="the current plan is for placement 'other'"):
        run(level(), [receipt()], current=current)


def test_the_estate_plan_and_report_are_byte_stable():
    plan_text, report_text = rendered(plan(**estate()))
    assert plan_text == (FIXTURES / "expected-plan.json").read_text(encoding="utf-8")
    assert report_text == (FIXTURES / "expected-report.json").read_text(encoding="utf-8")
    assert rendered(plan(**estate())) == (plan_text, report_text)


def test_input_order_does_not_change_the_output():
    baseline = rendered(plan(**estate()))
    shuffler = random.Random(11)  # noqa: S311 -- reproducible test data
    for _ in range(5):
        inputs = estate()
        shuffler.shuffle(inputs["receipts"])
        shuffler.shuffle(inputs["boot_evidence"])
        service_levels = inputs["service_levels"]
        inputs["service_levels"] = ServiceLevels(
            identities=tuple(reversed(service_levels.identities)), tiers=tuple(reversed(service_levels.tiers))
        )
        source = inputs["demand"]
        inputs["demand"] = Demand(start=source.start, end=source.end, series=tuple(reversed(source.series)))
        assert rendered(plan(**inputs)) == baseline


def test_replanning_the_plan_it_wrote_changes_nothing():
    first = plan(**estate())
    second = plan(**{**estate(), "current_plan": first.plan})
    assert second.plan == first.plan
    assert render(second.plan.to_dict()) == render(first.plan.to_dict())
    assert second.report.unchanged
    assert second.report.to_dict()["changes"] == {"identities": [], "lanes": []}


def test_the_estate_plan_parses_as_a_plan_document():
    result = plan(**estate())
    assert Plan.from_dict(json.loads(render(result.plan.to_dict()))) == result.plan


def test_the_estate_places_every_identity_on_a_measured_feasible_cell():
    inputs = estate()
    result = plan(**inputs)
    receipts = {document.source.uri: document for document in inputs["receipts"]}
    for entry in result.report.identities:
        if entry.status is not IdentityStatus.PLACED:
            continue
        chosen = entry.chosen
        assert chosen.blocked_by == ()
        measured = receipts[chosen.receipt.uri]
        assert measured.cell == chosen.cell
        if measured.latency is not None and entry.service_level.warm_p99_ms is not None:
            assert measured.latency.p99_ms <= entry.service_level.warm_p99_ms
        if measured.itl is not None:
            assert measured.ttft.p99_ms <= entry.service_level.ttft_p99_ms
            assert measured.itl.p99_ms <= entry.service_level.itl_p99_ms
        if chosen.warmth is not Warmth.PINNED:
            assert entry.service_level.come_alive_p95_s > 0
            assert chosen.come_alive_s <= entry.service_level.come_alive_p95_s


def test_the_estate_report_states_its_findings():
    report = plan(**estate()).report
    dense = report.identity("BAAI/bge-m3")
    assert (dense.chosen.cell, dense.chosen.warmth, dense.lane) == (L4, Warmth.PINNED, "default-l4-default")
    assert dense.chosen.monthly_cost == pytest.approx(1.04 * MONTH)
    assert dense.next_cheaper.warmth is Warmth.SNAPSHOT_COLD
    assert dense.next_cheaper.blocked_by == (Blocker(field="comeAliveP95S", bound=0, measured=8.5),)
    sparse = report.identity("BAAI/bge-m3:sparse")
    assert (sparse.chosen.warmth, sparse.chosen.come_alive_s) == (Warmth.COLD_WITH_BOUND, 95.0)
    assert sparse.chosen.monthly_cost == pytest.approx(1.04 * (3 / 40 + 3 * 2 * 900) / 3600 * SCALE)
    generation = report.identity("Qwen/Qwen3-8B")
    assert (generation.chosen.cell, generation.chosen.warmth) == (A100, Warmth.COLD_WITH_BOUND)
    assert generation.chosen.monthly_cost == pytest.approx(3.14 * 2 * 2 * SCALE)
    assert generation.next_cheaper.blocked_by == (Blocker(field="itlP99Ms", bound=80, measured=118.0),)
    assert report.identity("Qwen/Qwen3-Reranker-0.6B").status is IdentityStatus.UNPLACED
    kept = report.identity("intfloat/e5-base-v2")
    assert (kept.status, kept.lane, kept.monthly_cost) == (IdentityStatus.KEPT, "default-l4-default", None)
    assert [refusal.record for refusal in report.refusals] == [RefusedRecord.BOOT_EVIDENCE, RefusedRecord.RECEIPT]
    assert report.queue_wait_unattained == ("BAAI/bge-m3", "BAAI/bge-m3:sparse", "intfloat/e5-base-v2")
    assert [change.identity for change in report.identity_changes] == ["BAAI/bge-m3:sparse", "Qwen/Qwen3-Embedding-4B"]
    assert report.monthly_cost == pytest.approx(sum(lane.monthly_cost for lane in report.lanes))
    shared = {lane.name: lane for lane in report.lanes}["default-l4-default"]
    assert shared.identities == ("BAAI/bge-m3", "intfloat/e5-base-v2")
    assert shared.monthly_cost == pytest.approx(dense.monthly_cost)
    assert shared.unpriced_identities == ("intfloat/e5-base-v2",)
    assert report.unpriced_lanes == ("default-l4-default",)
    (chart_lane,) = report.shared_chart_lanes
    assert chart_lane.lanes == ("baai-bge-m3-sparse-l4", "default-l4-default", "qwen-qwen3-embedding-4b-l4")
    current = estate()["current_plan"]
    assert sum(lane.envelope.floor_min for lane in plan(**estate()).plan.lanes) == 1 + sum(
        lane.envelope.floor_min for lane in current.lanes
    )


def test_an_estate_lever_saves_what_planning_again_at_its_permitted_bound_saves():
    inputs = estate()
    result = plan(**inputs)
    report = result.report
    assert [(lever.identity, lever.warmth) for lever in report.come_alive_levers] == [
        ("Qwen/Qwen3-Embedding-4B", Warmth.COLD_WITH_BOUND),
        ("Qwen/Qwen3-Embedding-4B", Warmth.SNAPSHOT_COLD),
    ]
    for lever in report.come_alive_levers:
        service_levels = loosened(inputs["service_levels"], lever.identity, lever.permitted_bound_s)
        again = plan(**{**inputs, "service_levels": service_levels, "current_plan": result.plan})
        assert report.monthly_cost - again.report.monthly_cost == pytest.approx(lever.saving_at_permitted_bound)
        assert lever.saving_at_permitted_bound > 0


def test_an_estate_identity_beside_a_kept_neighbour_offers_no_lever_and_no_saving():
    inputs = estate()
    result = plan(**inputs)
    dense = result.report.identity("BAAI/bge-m3")
    assert (dense.lane, dense.chosen.warmth) == ("default-l4-default", Warmth.PINNED)
    assert dense.next_cheaper.come_alive_s == 8.5
    assert dense.chosen.monthly_cost > dense.next_cheaper.monthly_cost
    assert dense.next_cheaper.saving == 0
    assert not [lever for lever in result.report.come_alive_levers if lever.identity == "BAAI/bge-m3"]
    for bound in (9, 95):
        service_levels = loosened(inputs["service_levels"], "BAAI/bge-m3", bound)
        again = plan(**{**inputs, "service_levels": service_levels, "current_plan": result.plan})
        assert again.report.identity("BAAI/bge-m3").chosen.warmth in (Warmth.COLD_WITH_BOUND, Warmth.SNAPSHOT_COLD)
        assert again.report.monthly_cost > result.report.monthly_cost


def test_an_estate_next_cheaper_saving_is_what_planning_again_with_its_bounds_loosened_saves():
    inputs = estate()
    result = plan(**inputs)
    offered = [entry for entry in result.report.identities if entry.next_cheaper is not None]
    assert [entry.identity for entry in offered] == ["BAAI/bge-m3", "Qwen/Qwen3-8B", "Qwen/Qwen3-Embedding-4B"]
    for entry in offered:
        service_levels = relaxed(inputs["service_levels"], entry.identity, entry.next_cheaper.blocked_by)
        again = plan(**{**inputs, "service_levels": service_levels, "current_plan": result.plan})
        fall = result.report.monthly_cost - again.report.monthly_cost
        assert entry.next_cheaper.saving == pytest.approx(max(0.0, fall), abs=0.01)


@pytest.mark.parametrize("seed", range(25))
def test_random_estates_choose_only_feasible_cells_and_replan_to_themselves(seed):
    rng = random.Random(seed)  # noqa: S311 -- reproducible test data
    identities = [f"org/model-{index}" for index in range(rng.randint(1, 5))]
    service_levels = levels(
        *(
            (
                identity,
                ServiceLevel(
                    come_alive_p95_s=rng.choice([None, 0, 30, 120, 600]),
                    warm_p99_ms=rng.choice([None, 150, 250, 400]),
                ),
            )
            for identity in identities
        )
    )
    receipts = [
        receipt(
            identity=identity,
            cell=cell,
            concurrency=rng.choice([4, 8]),
            rps=rng.choice([0.5, 5.0, 50.0]),
            p99=rng.choice([100.0, 200.0, 300.0, 500.0]),
            uri=f"receipts/{identity}/{cell.machine_profile}/{index}",
        )
        for identity in identities
        for index, cell in enumerate([L4, A100, L4])
        if rng.random() < 0.7
    ]
    boots = [
        boot(
            identity=identity,
            cell=cell,
            warmth=warmth,
            seconds=rng.choice([5.0, 60.0, 200.0]),
            outcome=rng.choice([BootOutcome.SUCCEEDED, BootOutcome.SUCCEEDED, BootOutcome.FAILED]),
            borrowed=rng.random() < 0.3,
            uri=f"boots/{identity}/{cell.machine_profile}/{warmth.value}/{index}",
        )
        for identity in identities
        for cell in (L4, A100)
        for warmth in (Warmth.COLD_WITH_BOUND, Warmth.SNAPSHOT_COLD)
        for index in range(2)
        if rng.random() < 0.4
    ]
    hourly = {
        identity: {hour: rng.choice([0, 1, 50, 20000]) for hour in rng.sample(range(WINDOW_HOURS), rng.randint(0, 6))}
        for identity in identities
    }
    prices = price_list(l4=rng.choice([0.5, 1.0]), a100=rng.choice([2.0, 3.0]), core=0.1)
    overrides = {
        "buffer_containers": rng.randint(0, 2),
        "fallback_floor": rng.randint(0, 3),
        "cooldown_s": rng.choice([0, 60, 900]),
    }
    result = run(service_levels, receipts, boots, prices=prices, hourly=hourly, **overrides)
    assert Plan.from_dict(result.plan.to_dict()) == result.plan
    by_uri = {document.source.uri: document for document in receipts}
    for entry in result.report.identities:
        if entry.chosen is None:
            continue
        chosen = entry.chosen
        assert chosen.blocked_by == ()
        bound = entry.service_level.warm_p99_ms
        assert bound is None or by_uri[chosen.receipt.uri].latency.p99_ms <= bound
        come_alive = entry.service_level.come_alive_p95_s
        if chosen.warmth is not Warmth.PINNED:
            assert come_alive is None or come_alive > 0
            assert come_alive is None or chosen.come_alive_s <= come_alive
        if entry.next_cheaper is not None:
            assert entry.next_cheaper.blocked_by
            assert entry.next_cheaper.monthly_cost < chosen.monthly_cost
            looser = relaxed(service_levels, entry.identity, entry.next_cheaper.blocked_by)
            again = run(looser, receipts, boots, prices=prices, hourly=hourly, current=result.plan, **overrides)
            fall = result.report.monthly_cost - again.report.monthly_cost
            assert entry.next_cheaper.saving == pytest.approx(max(0.0, fall), abs=0.01)
    costs = {entry.identity: entry.chosen.monthly_cost for entry in result.report.identities if entry.chosen}
    for lever in result.report.come_alive_levers:
        assert lever.monthly_cost < costs[lever.identity]
        assert lever.saving_at_permitted_bound >= lever.saving_under_bound >= 0
        assert lever.saving_at_permitted_bound > 0
        looser = loosened(service_levels, lever.identity, lever.permitted_bound_s)
        again = run(looser, receipts, boots, prices=prices, hourly=hourly, current=result.plan, **overrides)
        assert result.report.monthly_cost - again.report.monthly_cost == pytest.approx(lever.saving_at_permitted_bound)
        if lever.bound_s == 0:
            assert lever.saving_under_bound == 0
            continue
        key = (lever.identity, lever.cell, lever.container_shape, lever.warmth)
        within = [
            replace(record, boot_seconds=min(record.boot_seconds, lever.bound_s))
            if (record.identity, record.cell, record.container_shape, record.warmth) == key
            else record
            for record in boots
        ]
        booted = run(service_levels, receipts, within, prices=prices, hourly=hourly, current=result.plan, **overrides)
        fall = result.report.monthly_cost - booted.report.monthly_cost
        assert lever.saving_under_bound == pytest.approx(max(0.0, fall), abs=0.01)
    for request in result.report.measurement_requests:
        assert request.saving_bound is None or request.saving_bound > 0
    again = run(service_levels, receipts, boots, prices=prices, hourly=hourly, current=result.plan, **overrides)
    assert again.plan == result.plan
    assert again.report.unchanged
    shuffled = list(receipts)
    rng.shuffle(shuffled)
    reordered = run(service_levels, shuffled, list(reversed(boots)), prices=prices, hourly=hourly, **overrides)
    assert rendered(reordered) == rendered(result)
