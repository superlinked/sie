from __future__ import annotations

import hashlib
import json
import math
import random
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


def single_lane_plan(name="shared", warmth=Warmth.PINNED, cell=L4, identity=IDENTITY) -> Plan:
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
                concurrency=8,
                max_containers=2,
                envelope=Envelope(floor_min=floor, floor_max=max(floor, 1)),
            ),
        ),
        identities=(PlanIdentity(identity=identity, lane=name, warmth=warmth),),
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
    assert (dense.chosen.cell, dense.chosen.warmth) == (L4, Warmth.PINNED)
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
    assert report.identity("intfloat/e5-base-v2").status is IdentityStatus.KEPT
    assert [refusal.record for refusal in report.refusals] == [RefusedRecord.BOOT_EVIDENCE, RefusedRecord.RECEIPT]
    assert report.queue_wait_unattained == ("BAAI/bge-m3", "BAAI/bge-m3:sparse", "intfloat/e5-base-v2")
    assert [change.identity for change in report.identity_changes] == [
        "BAAI/bge-m3",
        "BAAI/bge-m3:sparse",
        "Qwen/Qwen3-Embedding-4B",
    ]
    assert report.monthly_cost == pytest.approx(
        sum(entry.chosen.monthly_cost for entry in report.identities if entry.chosen is not None)
    )


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
    costs = {entry.identity: entry.chosen.monthly_cost for entry in result.report.identities if entry.chosen}
    for lever in result.report.come_alive_levers:
        assert lever.monthly_cost < costs[lever.identity]
        assert lever.saving_at_permitted_bound >= lever.saving_under_bound >= 0
    for request in result.report.measurement_requests:
        assert request.saving_bound is None or request.saving_bound > 0
    again = run(service_levels, receipts, boots, prices=prices, hourly=hourly, current=result.plan, **overrides)
    assert again.plan == result.plan
    assert again.report.unchanged
    shuffled = list(receipts)
    rng.shuffle(shuffled)
    reordered = run(service_levels, shuffled, list(reversed(boots)), prices=prices, hourly=hourly, **overrides)
    assert rendered(reordered) == rendered(result)
