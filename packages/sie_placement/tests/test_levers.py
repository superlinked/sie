from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sie_placement.planner import PlannerParameters, plan
from sie_placement.report import ComeAliveBasis
from sie_placement.schema import (
    BootEvidence,
    BootOutcome,
    Cell,
    ContainerShape,
    Demand,
    DemandHour,
    DemandSeries,
    IdentityServiceLevel,
    MachineProfilePrice,
    Operation,
    Percentiles,
    Placement,
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
SCALE = MONTH / 24
L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
WIDE = ContainerShape(gpu_count=1, cpu_cores=8, memory_mib=32768)
PRICES = PriceList(
    currency="USD",
    machine_profiles=(
        MachineProfilePrice(
            name="l4", gpu_product="NVIDIA-L4", accelerator_per_hour=1.0, core_per_hour=0.1, memory_gib_per_hour=0.0
        ),
        MachineProfilePrice(
            name="a100-80gb",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            accelerator_per_hour=3.0,
            core_per_hour=0.1,
            memory_gib_per_hour=0.0,
        ),
    ),
    placements=(Placement(name="test", region_multiplier=1),),
)
PARAMETERS = PlannerParameters(placement="test", max_containers=4, fallback_floor=2, cooldown_s=900)
L4_HOURLY = 1.0 + 4 * 0.1


def receipt(identity, cell=L4, *, shape=SHAPE, p99=100.0, concurrency=8):
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=shape,
        concurrency=concurrency,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=10.0),
        latency=Percentiles(p99_ms=p99),
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}/{shape.memory_mib}/{concurrency}"),
    )


def boot(identity, seconds, warmth=Warmth.COLD_WITH_BOUND, cell=L4, *, shape=SHAPE, borrowed=False):
    return BootEvidence(
        identity=identity,
        cell=cell,
        container_shape=shape,
        warmth=warmth,
        boot_seconds=seconds,
        outcome=BootOutcome.SUCCEEDED,
        borrowed=borrowed,
        source=SourceRef(uri=f"boots/{identity}/{cell.machine_profile}/{warmth.value}"),
    )


def busy(*identities, hours=6):
    return Demand(
        start=START,
        end=START + timedelta(hours=24),
        series=tuple(
            DemandSeries(
                identity=identity,
                operation=Operation.ENCODE,
                hours=tuple(DemandHour(start=START + timedelta(hours=hour), requests=10) for hour in range(hours)),
            )
            for identity in identities
        ),
    )


def run(bounds, receipts, boots, demand=None):
    service_levels = ServiceLevels(
        identities=tuple(
            IdentityServiceLevel(identity=identity, level=ServiceLevel(come_alive_p95_s=bound, warm_p99_ms=250))
            for identity, bound in bounds.items()
        )
    )
    return plan(service_levels, receipts, boots, PRICES, demand or busy(*bounds), None, PARAMETERS).report


def cold_cost(hours=6):
    return L4_HOURLY * hours * 2 * SCALE


def test_a_zero_bound_lever_saves_nothing_under_its_bound_and_is_priced_at_the_bound_it_permits():
    report = run({"org/warm": 0}, [receipt("org/warm")], [boot("org/warm", 95.2, borrowed=True)])
    (lever,) = report.come_alive_levers
    assert (lever.identity, lever.cell, lever.container_shape, lever.warmth) == (
        "org/warm",
        L4,
        SHAPE,
        Warmth.COLD_WITH_BOUND,
    )
    assert (lever.bound_s, lever.come_alive_s, lever.permitted_bound_s) == (0, 95.2, 96)
    assert lever.come_alive_basis is ComeAliveBasis.BORROWED_BOOT_EVIDENCE
    assert lever.saving_under_bound == 0
    assert lever.monthly_cost == pytest.approx(cold_cost())
    assert lever.saving_at_permitted_bound == pytest.approx(L4_HOURLY * MONTH - cold_cost())


def test_a_finite_bound_lever_saves_the_same_under_its_bound_and_at_the_bound_it_permits():
    report = run({"org/slow": 60}, [receipt("org/slow")], [boot("org/slow", 240.0)])
    (lever,) = report.come_alive_levers
    assert (lever.bound_s, lever.permitted_bound_s) == (60, 240)
    assert lever.saving_under_bound == pytest.approx(L4_HOURLY * MONTH - cold_cost())
    assert lever.saving_at_permitted_bound == lever.saving_under_bound


@pytest.mark.parametrize("seconds", [0.0, 0.4])
def test_a_sub_second_restore_permits_a_one_second_bound(seconds):
    report = run({"org/warm": 0}, [receipt("org/warm")], [boot("org/warm", seconds, Warmth.SNAPSHOT_COLD)])
    assert report.come_alive_levers[0].permitted_bound_s == 1


def test_levers_rank_by_the_saving_under_the_bound_then_at_the_permitted_bound():
    bounds = {"org/always-warm": 0, "org/small-saving": 60, "org/big-saving": 60, "org/zero-quiet": 0}
    receipts = [receipt(identity) for identity in bounds]
    boots = [boot(identity, 120.0) for identity in bounds]
    demand = Demand(
        start=START,
        end=START + timedelta(hours=24),
        series=(
            *busy("org/small-saving", hours=8).series,
            *busy("org/big-saving", hours=2).series,
            *busy("org/always-warm", hours=1).series,
            *busy("org/zero-quiet", hours=4).series,
        ),
    )
    report = run(bounds, receipts, boots, demand)
    ranked = [(lever.identity, lever.saving_under_bound > 0) for lever in report.come_alive_levers]
    assert ranked == [
        ("org/big-saving", True),
        ("org/small-saving", True),
        ("org/always-warm", False),
        ("org/zero-quiet", False),
    ]
    savings = [lever.saving_at_permitted_bound for lever in report.come_alive_levers]
    assert savings[2] > savings[3]


def test_a_cell_that_misses_its_warm_bound_offers_no_lever():
    report = run(
        {"org/model": 60}, [receipt("org/model", p99=400.0), receipt("org/model", A100)], [boot("org/model", 240.0)]
    )
    assert report.come_alive_levers == ()
    assert report.identity("org/model").chosen.cell == A100


def test_a_cold_posture_that_is_not_cheaper_offers_no_lever():
    report = run({"org/model": 60}, [receipt("org/model")], [boot("org/model", 240.0)], busy("org/model", hours=24))
    assert report.identity("org/model").chosen.warmth is Warmth.PINNED
    assert report.come_alive_levers == ()


def test_an_undeclared_bound_offers_no_lever():
    report = run({"org/model": None}, [receipt("org/model")], [boot("org/model", 240.0)])
    assert report.identity("org/model").chosen.warmth is Warmth.COLD_WITH_BOUND
    assert report.come_alive_levers == ()


def test_a_measured_come_alive_under_the_bound_is_a_choice_and_not_a_lever():
    report = run({"org/model": 60}, [receipt("org/model")], [boot("org/model", 30.0)])
    assert report.identity("org/model").chosen.warmth is Warmth.COLD_WITH_BOUND
    assert report.come_alive_levers == ()


def test_each_cell_shape_and_warmth_is_one_lever_at_its_cheapest_receipt():
    receipts = [
        receipt("org/model", concurrency=8),
        receipt("org/model", concurrency=16),
        receipt("org/model", shape=WIDE),
    ]
    boots = [
        boot("org/model", 240.0),
        boot("org/model", 30.0 + 60.0, Warmth.SNAPSHOT_COLD),
        boot("org/model", 300.0, shape=WIDE),
    ]
    report = run({"org/model": 60}, receipts, boots)
    keys = [(lever.container_shape, lever.warmth) for lever in report.come_alive_levers]
    assert sorted(keys, key=str) == sorted(
        [(SHAPE, Warmth.COLD_WITH_BOUND), (SHAPE, Warmth.SNAPSHOT_COLD), (WIDE, Warmth.COLD_WITH_BOUND)], key=str
    )
    assert len(keys) == len(set(keys))
    by_key = {(lever.container_shape, lever.warmth): lever for lever in report.come_alive_levers}
    assert by_key[(SHAPE, Warmth.COLD_WITH_BOUND)].monthly_cost == pytest.approx(cold_cost())
    assert by_key[(WIDE, Warmth.COLD_WITH_BOUND)].monthly_cost == pytest.approx((1.0 + 8 * 0.1) * 6 * 2 * SCALE)
    assert by_key[(SHAPE, Warmth.SNAPSHOT_COLD)].permitted_bound_s == 90
