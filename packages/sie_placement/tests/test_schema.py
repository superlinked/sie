from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sie_placement.schema import (
    BootEvidence,
    Capacity,
    Cell,
    ContainerShape,
    Demand,
    DemandHour,
    DemandSeries,
    Envelope,
    IdentityServiceLevel,
    Operation,
    Percentiles,
    Plan,
    PlanIdentity,
    PlanLane,
    PriceList,
    Receipt,
    SchemaError,
    ServiceLevel,
    ServiceLevels,
    SharedQuota,
    TypeCeiling,
    Warmth,
)

FIXTURES = Path(__file__).parent / "fixtures"

DOCUMENTS: list[tuple[str, Callable[..., object]]] = [
    ("service-levels.json", ServiceLevels.from_dict),
    ("receipt-encode.json", Receipt.from_dict),
    ("receipt-generate.json", Receipt.from_dict),
    ("boot-evidence.json", BootEvidence.from_dict),
    ("price-list.json", PriceList.from_dict),
    ("demand.json", Demand.from_dict),
    ("capacity.json", Capacity.from_dict),
    ("plan.json", Plan.from_dict),
]


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def schema_error(parse: Callable[[dict], object], data: dict) -> SchemaError:
    with pytest.raises(SchemaError) as caught:
        parse(data)
    return caught.value


@pytest.mark.parametrize(("name", "parse"), DOCUMENTS)
def test_canonical_documents_round_trip(name, parse):
    data = load(name)
    document = parse(data)
    assert document.to_dict() == data
    assert json.dumps(document.to_dict()) == json.dumps(data)
    assert parse(json.loads(json.dumps(document.to_dict()))) == document


@pytest.mark.parametrize(("name", "parse"), DOCUMENTS)
def test_documents_reject_another_kind(name, parse):
    data = load(name)
    data["kind"] = "placement-other"
    error = schema_error(parse, data)
    assert error.path == "$.kind"
    assert "expected 'placement-" in error.message


@pytest.mark.parametrize(("name", "parse"), DOCUMENTS)
def test_documents_reject_unsupported_versions(name, parse):
    data = load(name)
    data["schemaVersion"] = 2
    error = schema_error(parse, data)
    assert str(error) == "$.schemaVersion: unsupported version 2; this reader supports version 1"


@pytest.mark.parametrize(("name", "parse"), DOCUMENTS)
def test_documents_require_a_version(name, parse):
    data = load(name)
    del data["schemaVersion"]
    assert str(schema_error(parse, data)) == "$.schemaVersion: is required"


@pytest.mark.parametrize(("name", "parse"), DOCUMENTS)
def test_documents_reject_unknown_top_level_fields(name, parse):
    data = load(name)
    data["notes"] = "free text"
    assert str(schema_error(parse, data)) == "$.notes: is not a field of this object"


@pytest.mark.parametrize("value", [[], "receipt", 1, None])
def test_documents_must_be_objects(value):
    error = schema_error(Receipt.from_dict, value)
    assert error.path == "$"
    assert error.message.startswith("expected an object, got ")


def test_nested_unknown_fields_are_reported_at_their_path():
    data = load("receipt-encode.json")
    data["cell"]["gpuType"] = "nvidia-l4"
    assert str(schema_error(Receipt.from_dict, data)) == "$.cell.gpuType: is not a field of this object"


def test_missing_nested_fields_are_reported_at_their_path():
    data = load("plan.json")
    del data["lanes"][1]["containerShape"]["memoryMiB"]
    assert str(schema_error(Plan.from_dict, data)) == "$.lanes[1].containerShape.memoryMiB: is required"


def test_null_is_never_a_value():
    data = load("receipt-encode.json")
    data["latency"]["p50Ms"] = None
    assert str(schema_error(Receipt.from_dict, data)) == "$.latency.p50Ms: must be omitted rather than null"
    data = load("receipt-encode.json")
    data["concurrency"] = None
    assert str(schema_error(Receipt.from_dict, data)) == "$.concurrency: must not be null"


@pytest.mark.parametrize(("value", "described"), [(True, "boolean true"), (16.0, "number 16.0"), ("16", "string '16'")])
def test_integers_are_strict(value, described):
    data = load("receipt-encode.json")
    data["concurrency"] = value
    assert str(schema_error(Receipt.from_dict, data)) == f"$.concurrency: expected an integer, got {described}"


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, json.loads("1" + "0" * 400), -(10**400)],
    ids=["nan", "infinity", "integer-above-float-range", "integer-below-float-range"],
)
def test_numbers_must_be_finite(value):
    data = load("receipt-encode.json")
    data["throughput"]["requestsPerSecond"] = value
    error = schema_error(Receipt.from_dict, data)
    assert error.path == "$.throughput.requestsPerSecond"
    assert error.message.startswith("expected a finite number")


def test_numbers_reject_booleans():
    data = load("price-list.json")
    data["placements"][0]["regionMultiplier"] = False
    assert (
        str(schema_error(PriceList.from_dict, data))
        == "$.placements[0].regionMultiplier: expected a number, got boolean false"
    )


def test_enumerations_list_their_choices():
    data = load("plan.json")
    data["identities"][0]["warmth"] = "warm"
    error = schema_error(Plan.from_dict, data)
    assert error.path == "$.identities[0].warmth"
    assert error.message == (
        "expected one of 'pinned', 'cold-with-bound', 'bridged', 'snapshot-cold', 'scheduled-warm', "
        "'lazy-resident', got 'warm'"
    )


def test_arrays_are_required_where_declared():
    data = load("plan.json")
    data["lanes"] = {"default-l4-default": {}}
    assert str(schema_error(Plan.from_dict, data)) == "$.lanes: expected an array, got an object"


@pytest.mark.parametrize("value", ["", "two words", "trailing\n"])
def test_identifiers_reject_empty_and_whitespace(value):
    data = load("receipt-encode.json")
    data["identity"] = value
    error = schema_error(Receipt.from_dict, data)
    assert error.path == "$.identity"
    assert error.message.startswith("must be a non-empty string without whitespace")


def test_service_levels_resolve_tier_defaults_and_overrides():
    levels = ServiceLevels.from_dict(load("service-levels.json"))
    assert levels.effective("BAAI/bge-m3") == ServiceLevel(come_alive_p95_s=0, warm_p99_ms=250, queue_wait_p99_ms=500)
    assert levels.effective("BAAI/bge-m3:sparse").come_alive_p95_s == 120
    generation = levels.effective("Qwen/Qwen3-8B")
    assert generation.is_generation
    assert (generation.ttft_p99_ms, generation.itl_p99_ms) == (4000, 80)
    assert levels.effective("Qwen/Qwen3-Reranker-0.6B") == ServiceLevel(come_alive_p95_s=90, warm_p99_ms=400)
    with pytest.raises(KeyError):
        levels.effective("unknown/model")


def test_service_levels_omit_an_empty_tier_list():
    data = load("service-levels.json")
    data["tiers"] = []
    data["identities"] = [{"identity": "BAAI/bge-m3", "warmP99Ms": 200}]
    assert ServiceLevels.from_dict(data).to_dict() == {
        "kind": "placement-service-levels",
        "schemaVersion": 1,
        "identities": [{"identity": "BAAI/bge-m3", "warmP99Ms": 200}],
    }


def test_generation_identities_do_not_declare_a_warm_latency():
    data = load("service-levels.json")
    data["identities"][3] = {"identity": "Qwen/Qwen3-8B-Chat", "warmP99Ms": 300, "ttftP99Ms": 900, "itlP99Ms": 60}
    error = schema_error(ServiceLevels.from_dict, data)
    assert error.path == "$.identities[3].warmP99Ms"
    assert "generation identity" in error.message


def test_a_generation_override_cannot_join_a_warm_latency_tier():
    data = load("service-levels.json")
    data["identities"][0]["ttftP99Ms"] = 900
    data["identities"][0]["itlP99Ms"] = 60
    error = schema_error(ServiceLevels.from_dict, data)
    assert error.path == "$.identities[0].warmP99Ms"


def test_generation_latencies_are_declared_together():
    data = load("service-levels.json")
    data["identities"][3] = {"identity": "Qwen/Qwen3-8B-Chat", "ttftP99Ms": 900}
    assert (
        str(schema_error(ServiceLevels.from_dict, data))
        == "$.identities[3].itlP99Ms: ttftP99Ms and itlP99Ms are declared together"
    )
    data = load("service-levels.json")
    del data["tiers"][1]["itlP99Ms"]
    assert (
        str(schema_error(ServiceLevels.from_dict, data))
        == "$.tiers[1].itlP99Ms: ttftP99Ms and itlP99Ms are declared together"
    )


def test_service_level_tiers_must_exist_and_names_must_be_unique():
    data = load("service-levels.json")
    data["identities"][0]["tier"] = "realtime"
    assert (
        str(schema_error(ServiceLevels.from_dict, data)) == "$.identities[0].tier: names no declared tier: 'realtime'"
    )
    data = load("service-levels.json")
    data["identities"][2]["identity"] = "BAAI/bge-m3"
    assert str(schema_error(ServiceLevels.from_dict, data)) == "$.identities[2].identity: duplicates 'BAAI/bge-m3'"
    data = load("service-levels.json")
    data["tiers"][1]["name"] = "interactive-retrieval"
    assert str(schema_error(ServiceLevels.from_dict, data)) == "$.tiers[1].name: duplicates 'interactive-retrieval'"


def test_service_level_bounds_are_non_negative():
    data = load("service-levels.json")
    data["identities"][3]["comeAliveP95S"] = -1
    assert (
        str(schema_error(ServiceLevels.from_dict, data)) == "$.identities[3].comeAliveP95S: must be at least 0, got -1"
    )


def test_generate_receipts_report_ttft_and_itl():
    data = load("receipt-generate.json")
    data["latency"] = data.pop("ttft")
    error = schema_error(Receipt.from_dict, data)
    assert error.path == "$.latency"
    assert error.message.startswith("must be omitted for the generate operation")
    data = load("receipt-generate.json")
    del data["itl"]
    assert str(schema_error(Receipt.from_dict, data)) == "$.itl: is required for the generate operation"


def test_other_receipts_report_one_latency():
    data = load("receipt-encode.json")
    data["ttft"] = data.pop("latency")
    assert str(schema_error(Receipt.from_dict, data)) == "$.latency: is required for an operation other than generate"
    data = load("receipt-encode.json")
    data["throughput"]["outputTokensPerSecond"] = 10
    error = schema_error(Receipt.from_dict, data)
    assert error.path == "$.throughput.outputTokensPerSecond"


def test_request_shapes_tie_output_tokens_to_generation():
    data = load("receipt-generate.json")
    del data["requestShape"]["outputTokens"]
    assert (
        str(schema_error(Receipt.from_dict, data))
        == "$.requestShape.outputTokens: is required for the generate operation"
    )
    data = load("receipt-encode.json")
    data["requestShape"]["outputTokens"] = 32
    error = schema_error(Receipt.from_dict, data)
    assert error.path == "$.requestShape.outputTokens"
    assert error.message.startswith("must be omitted")


def test_receipt_shapes_and_rates_are_bounded():
    data = load("receipt-encode.json")
    data["containerShape"]["gpuCount"] = 0
    assert str(schema_error(Receipt.from_dict, data)) == "$.containerShape.gpuCount: must be at least 1, got 0"
    data = load("receipt-encode.json")
    data["throughput"]["requestsPerSecond"] = 0
    assert str(schema_error(Receipt.from_dict, data)) == "$.throughput.requestsPerSecond: must be greater than 0, got 0"
    data = load("receipt-encode.json")
    data["concurrency"] = 0
    assert str(schema_error(Receipt.from_dict, data)) == "$.concurrency: must be at least 1, got 0"


def test_percentiles_are_ordered():
    data = load("receipt-encode.json")
    data["latency"]["p95Ms"] = 300
    assert str(schema_error(Receipt.from_dict, data)) == "$.latency.p99Ms: must be at least p95Ms (300), got 214"
    data = load("receipt-generate.json")
    data["ttft"]["p50Ms"] = 2000
    assert str(schema_error(Receipt.from_dict, data)) == "$.ttft.p99Ms: must be at least p50Ms (2000), got 1480"


def test_source_digests_are_sha256_hex():
    data = load("receipt-encode.json")
    data["source"]["sha256"] = data["source"]["sha256"].upper()
    error = schema_error(Receipt.from_dict, data)
    assert error.path == "$.source.sha256"


def test_boot_evidence_records_only_booting_warmths():
    data = load("boot-evidence.json")
    data["warmth"] = "pinned"
    assert str(schema_error(BootEvidence.from_dict, data)) == (
        "$.warmth: boot evidence records one of 'cold-with-bound', 'snapshot-cold', 'lazy-resident', got 'pinned'"
    )


def test_boot_evidence_keeps_failed_outcomes():
    data = load("boot-evidence.json")
    data["outcome"] = "timed-out"
    data["bootSeconds"] = 900
    evidence = BootEvidence.from_dict(data)
    assert evidence.outcome.value == "timed-out"
    assert evidence.to_dict() == data
    data["bootSeconds"] = -0.5
    assert str(schema_error(BootEvidence.from_dict, data)) == "$.bootSeconds: must be at least 0, got -0.5"


def test_boot_evidence_marks_evidence_borrowed_from_another_environment():
    data = load("boot-evidence.json")
    assert BootEvidence.from_dict(data).borrowed is False
    data["borrowed"] = True
    evidence = BootEvidence.from_dict(data)
    assert evidence.borrowed is True
    assert evidence.to_dict() == data
    data["borrowed"] = False
    evidence = BootEvidence.from_dict(data)
    assert evidence.borrowed is False
    assert "borrowed" not in evidence.to_dict()


@pytest.mark.parametrize(("value", "described"), [(1, "number 1"), ("true", "string 'true'")])
def test_boot_evidence_borrowed_is_a_strict_boolean(value, described):
    data = load("boot-evidence.json")
    data["borrowed"] = value
    assert str(schema_error(BootEvidence.from_dict, data)) == f"$.borrowed: expected a boolean, got {described}"


def test_boot_evidence_records_an_unplaced_attempt_without_a_boot():
    data = load("boot-evidence.json")
    data["outcome"] = "unplaced"
    del data["bootSeconds"]
    evidence = BootEvidence.from_dict(data)
    assert (evidence.outcome.value, evidence.boot_seconds, evidence.placement_wait_s) == ("unplaced", None, 41.2)
    assert evidence.to_dict() == data
    del data["placementWaitS"]
    assert BootEvidence.from_dict(data).to_dict() == data
    data["bootSeconds"] = 30
    assert str(schema_error(BootEvidence.from_dict, data)) == (
        "$.bootSeconds: must be omitted for an unplaced attempt, which never booted"
    )


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "timed-out"])
def test_boot_evidence_of_a_placed_attempt_records_its_boot(outcome):
    data = load("boot-evidence.json")
    data["outcome"] = outcome
    del data["bootSeconds"]
    assert str(schema_error(BootEvidence.from_dict, data)) == (
        "$.bootSeconds: is required for an attempt that was placed"
    )


def test_boot_evidence_placement_waits_are_optional_and_non_negative():
    data = load("boot-evidence.json")
    del data["placementWaitS"]
    evidence = BootEvidence.from_dict(data)
    assert evidence.placement_wait_s is None
    assert "placementWaitS" not in evidence.to_dict()
    data["placementWaitS"] = -1
    assert str(schema_error(BootEvidence.from_dict, data)) == "$.placementWaitS: must be at least 0, got -1"


def test_a_lazy_resident_load_runs_on_a_placed_container():
    data = load("boot-evidence.json")
    data["warmth"] = "lazy-resident"
    assert str(schema_error(BootEvidence.from_dict, data)) == (
        "$.placementWaitS: must be omitted for a lazy-resident load, which runs on a placed container"
    )
    del data["placementWaitS"]
    assert BootEvidence.from_dict(data).warmth is Warmth.LAZY_RESIDENT
    data["outcome"] = "unplaced"
    del data["bootSeconds"]
    assert str(schema_error(BootEvidence.from_dict, data)) == (
        "$.outcome: a lazy-resident load runs on a placed container and is never unplaced"
    )


def test_capacity_declares_a_total_type_ceilings_or_a_shared_quota():
    assert str(schema_error(Capacity.from_dict, {"kind": "placement-capacity", "schemaVersion": 1})) == (
        "$: must declare total, types or sharedQuota"
    )
    assert Capacity(total=0).to_dict() == {"kind": "placement-capacity", "schemaVersion": 1, "total": 0}
    quota = Capacity.from_dict(
        {"kind": "placement-capacity", "schemaVersion": 1, "sharedQuota": {"accelerators": 4}}
    ).shared_quota
    assert quota == SharedQuota(accelerators=4, held_elsewhere=0)
    assert quota.to_dict() == {"accelerators": 4, "heldElsewhere": 0}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(total=-1), "$.total: must be at least 0, got -1"),
        (lambda data: data["types"][1].update(accelerators=-2), "$.types[1].accelerators: must be at least 0, got -2"),
        (
            lambda data: data["types"][1].update(gpuProduct="NVIDIA-L4"),
            "$.types[1].gpuProduct: duplicates 'NVIDIA-L4'",
        ),
        (lambda data: data["types"][0].update(gpuProduct="NVIDIA L4"), "$.types[0].gpuProduct: must be a non-empty"),
        (
            lambda data: data["sharedQuota"].update(heldElsewhere=-1),
            "$.sharedQuota.heldElsewhere: must be at least 0, got -1",
        ),
        (lambda data: data["sharedQuota"].pop("accelerators"), "$.sharedQuota.accelerators: is required"),
        (lambda data: data["types"][0].update(max=3), "$.types[0].max: is not a field of this object"),
    ],
)
def test_capacity_ceilings_are_validated(mutate, message):
    data = load("capacity.json")
    mutate(data)
    assert str(schema_error(Capacity.from_dict, data)).startswith(message)


def test_capacity_built_in_code_names_type_ceilings_by_gpu_product():
    capacity = Capacity(types=(TypeCeiling(gpu_product="NVIDIA-L4", accelerators=2),))
    assert Capacity.from_dict(json.loads(json.dumps(capacity.to_dict()))) == capacity
    with pytest.raises(SchemaError) as caught:
        TypeCeiling(gpu_product="", accelerators=1)
    assert caught.value.path == "gpuProduct"


def test_price_lists_validate_currency_rates_and_names():
    data = load("price-list.json")
    data["currency"] = "usd"
    assert schema_error(PriceList.from_dict, data).path == "$.currency"
    data = load("price-list.json")
    data["machineProfiles"][1]["name"] = "l4"
    assert str(schema_error(PriceList.from_dict, data)) == "$.machineProfiles[1].name: duplicates 'l4'"
    data = load("price-list.json")
    data["machineProfiles"][0]["corePerHour"] = -0.01
    assert (
        str(schema_error(PriceList.from_dict, data))
        == "$.machineProfiles[0].corePerHour: must be at least 0, got -0.01"
    )
    data = load("price-list.json")
    data["placements"][1]["regionMultiplier"] = 0
    assert (
        str(schema_error(PriceList.from_dict, data))
        == "$.placements[1].regionMultiplier: must be greater than 0, got 0"
    )
    data = load("price-list.json")
    data["placements"] = []
    assert str(schema_error(PriceList.from_dict, data)) == "$.placements: must list at least one placement"


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-02T09:30:00Z",
        "2026-11-02T09:00:00+00:00",
        "2026-11-02 09:00:00Z",
        "2026-02-30T09:00:00Z",
        "٢٠٢٦-11-02T09:00:00Z",
    ],
)
def test_demand_hours_are_utc_hour_boundaries(value):
    data = load("demand.json")
    data["series"][0]["hours"][0]["start"] = value
    error = schema_error(Demand.from_dict, data)
    assert error.path == "$.series[0].hours[0].start"


def test_demand_hours_are_ordered_and_inside_the_window():
    data = load("demand.json")
    data["series"][0]["hours"].reverse()
    assert (
        str(schema_error(Demand.from_dict, data)) == "$.series[0].hours[1].start: must be later than the previous hour"
    )
    data = load("demand.json")
    data["series"][1]["hours"][0]["start"] = "2026-11-03T00:00:00Z"
    assert str(schema_error(Demand.from_dict, data)) == (
        "$.series[1].hours[0].start: must fall inside the window [2026-11-02T00:00:00Z, 2026-11-03T00:00:00Z)"
    )
    data = load("demand.json")
    data["end"] = data["start"]
    assert str(schema_error(Demand.from_dict, data)) == "$.end: must be later than start"


def test_demand_series_are_unique_per_identity_and_operation():
    data = load("demand.json")
    data["series"].append(copy.deepcopy(data["series"][0]))
    assert str(schema_error(Demand.from_dict, data)) == "$.series[2].identity: duplicates 'BAAI/bge-m3 encode'"
    data["series"][2]["operation"] = "score"
    assert len(Demand.from_dict(data).series) == 3


def test_demand_hours_built_in_code_must_be_utc_boundaries():
    with pytest.raises(SchemaError, match="UTC hour boundary"):
        DemandHour(start=datetime(2026, 11, 2, 9, 30, tzinfo=UTC), requests=1)
    with pytest.raises(SchemaError, match="UTC hour boundary"):
        DemandHour(start=datetime(2026, 11, 2, 9, tzinfo=timezone(timedelta(hours=1))), requests=1)
    with pytest.raises(SchemaError, match="UTC hour boundary"):
        DemandHour(start=datetime(2026, 11, 2, 9), requests=1)  # noqa: DTZ001
    hour = DemandHour(start=datetime(2026, 11, 2, 9, tzinfo=UTC), requests=4)
    series = DemandSeries(identity="BAAI/bge-m3", operation=Operation.ENCODE, hours=(hour,))
    demand = Demand(start=hour.start, end=hour.start + timedelta(hours=1), series=(series,))
    assert Demand.from_dict(demand.to_dict()) == demand


def test_demand_hours_serialize_four_digit_years():
    hour = DemandHour(start=datetime(999, 1, 1, 7, tzinfo=UTC), requests=1)
    assert hour.to_dict()["start"] == "0999-01-01T07:00:00Z"
    assert DemandHour.from_dict(hour.to_dict()) == hour


@pytest.mark.parametrize(
    ("envelope", "path", "message"),
    [
        ({"floorMin": 2, "floorMax": 1}, "floorMax", "must be at least floorMin (2), got 1"),
        ({"floorMin": -1, "floorMax": 1}, "floorMin", "must be at least 0, got -1"),
        ({"floorMin": 0, "floorMax": 1, "bufferMin": 0}, "bufferMax", "bufferMin and bufferMax are declared together"),
        (
            {"floorMin": 0, "floorMax": 1, "bufferMin": 2, "bufferMax": 1},
            "bufferMax",
            "must be at least bufferMin (2), got 1",
        ),
        (
            {"floorMin": 0, "floorMax": 1, "scaledownWindowMaxS": 60},
            "scaledownWindowMinS",
            "scaledownWindowMinS and scaledownWindowMaxS are declared together",
        ),
        (
            {"floorMin": 0, "floorMax": 1, "scaledownWindowMinS": 600, "scaledownWindowMaxS": 60},
            "scaledownWindowMaxS",
            "must be at least scaledownWindowMinS (600), got 60",
        ),
    ],
)
def test_envelopes_are_ordered_ranges(envelope, path, message):
    error = schema_error(Envelope.from_dict, envelope)
    assert error.path == f"$.{path}"
    assert error.message == message


def test_envelope_ranges_are_optional_for_floor_only_appliers():
    envelope = Envelope.from_dict({"floorMin": 0, "floorMax": 2})
    assert envelope.buffer_min is None
    assert envelope.scaledown_window_max_s is None
    assert envelope.to_dict() == {"floorMin": 0, "floorMax": 2}


def test_plan_envelopes_stay_under_the_lane_ceiling():
    data = load("plan.json")
    data["lanes"][0]["maxContainers"] = 2
    assert str(schema_error(Plan.from_dict, data)) == (
        "$.lanes[0].envelope.floorMax: must not exceed maxContainers (2), got 3"
    )


def test_plan_identities_name_a_lane_of_the_plan():
    data = load("plan.json")
    data["identities"][2]["lane"] = "default-h100-generation"
    assert str(schema_error(Plan.from_dict, data)) == (
        "$.identities[2].lane: names no lane of this plan: 'default-h100-generation'"
    )


def test_plan_lanes_host_an_identity():
    data = load("plan.json")
    del data["identities"][2]
    assert str(schema_error(Plan.from_dict, data)) == (
        "$.lanes[1].name: lane 'default-a100-80gb-generation' hosts no identity"
    )


@pytest.mark.parametrize(
    ("index", "warmth", "expectation"),
    [
        (0, "cold-with-bound", "floorMin 0, got 1"),
        (1, "snapshot-cold", "floorMin 0, got 1"),
        (2, "pinned", "floorMin at least 1, got 0"),
        (2, "lazy-resident", "floorMin at least 1, got 0"),
    ],
)
def test_plan_warmth_matches_the_lane_floor(index, warmth, expectation):
    data = load("plan.json")
    data["identities"][index]["warmth"] = warmth
    error = schema_error(Plan.from_dict, data)
    assert error.path == f"$.identities[{index}].warmth"
    assert error.message.startswith(f"{warmth} needs lane ")
    assert error.message.endswith(expectation)


@pytest.mark.parametrize("warmth", ["bridged", "scheduled-warm", "snapshot-cold"])
def test_plan_accepts_every_zero_floor_warmth(warmth):
    data = load("plan.json")
    data["identities"][2]["warmth"] = warmth
    assert Plan.from_dict(data).identities[2].warmth.value == warmth


def test_plan_lanes_name_fallback_cells_after_their_cell():
    data = load("plan.json")
    lane = Plan.from_dict(data).lanes[1]
    assert lane.cells == (
        Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB"),
        Cell(machine_profile="h100", gpu_product="NVIDIA-H100-80GB-HBM3"),
    )
    assert "fallbackCells" not in Plan.from_dict(data).lanes[0].to_dict()
    assert Plan.from_dict(data).lanes[0].cells == (Plan.from_dict(data).lanes[0].cell,)


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        (
            [{"machineProfile": "a100-80gb", "gpuProduct": "NVIDIA-A100-SXM4-80GB"}],
            "$.lanes[1].fallbackCells[0]: repeats 'a100-80gb' with 'NVIDIA-A100-SXM4-80GB', already a cell of this lane",
        ),
        (
            [
                {"machineProfile": "h100", "gpuProduct": "NVIDIA-H100-80GB-HBM3"},
                {"machineProfile": "h100", "gpuProduct": "NVIDIA-H100-80GB-HBM3"},
            ],
            "$.lanes[1].fallbackCells[1]: repeats 'h100' with 'NVIDIA-H100-80GB-HBM3', already a cell of this lane",
        ),
        ({"machineProfile": "h100"}, "$.lanes[1].fallbackCells: expected an array, got an object"),
    ],
)
def test_plan_lane_fallback_cells_are_distinct_from_each_other_and_the_cell(cells, message):
    data = load("plan.json")
    data["lanes"][1]["fallbackCells"] = cells
    assert str(schema_error(Plan.from_dict, data)) == message


def test_plan_lane_names_and_identities_are_unique():
    data = load("plan.json")
    data["lanes"][1]["name"] = "default-l4-default"
    assert str(schema_error(Plan.from_dict, data)) == "$.lanes[1].name: duplicates 'default-l4-default'"
    data = load("plan.json")
    data["identities"][1]["identity"] = "BAAI/bge-m3"
    assert str(schema_error(Plan.from_dict, data)) == "$.identities[1].identity: duplicates 'BAAI/bge-m3'"


def test_documents_built_in_code_report_relative_paths():
    with pytest.raises(SchemaError) as caught:
        Envelope(floor_min=3, floor_max=1)
    assert caught.value.path == "floorMax"
    with pytest.raises(SchemaError) as caught:
        IdentityServiceLevel(identity="model", level=ServiceLevel(ttft_p99_ms=1, warm_p99_ms=1))
    assert caught.value.path == "warmP99Ms"
    with pytest.raises(SchemaError) as caught:
        Percentiles(p50_ms=5, p99_ms=1)
    assert str(caught.value) == "p99Ms: must be at least p50Ms (5), got 1"


def test_plans_built_in_code_round_trip():
    cell = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
    shape = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
    lane = PlanLane(
        name="default-l4-default",
        queue_pool="default",
        bundle="default",
        cell=cell,
        container_shape=shape,
        concurrency=8,
        max_containers=4,
        envelope=Envelope(floor_min=1, floor_max=2),
    )
    plan = Plan(
        placement="us-central1",
        lanes=(lane,),
        identities=(PlanIdentity(identity="BAAI/bge-m3", lane=lane.name, warmth=Warmth.PINNED),),
    )
    assert Plan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
