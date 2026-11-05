from __future__ import annotations

import copy
import itertools
import json
import socketserver
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from sie_placement import keda
from sie_placement.schema import Plan
from sie_placement.tuner import Action

FIXTURES = Path(__file__).parent / "fixtures"
AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

QUEUE_METRIC = "sie_gateway_lane_queue_depth"
PENDING_METRIC = "sie_gateway_pending_demand"
FLOOR_METRIC = "sie_gateway_pool_warm_floor"
SLOTS_METRIC = "sie_worker_gpu_slots"


def plan_document() -> dict:
    return json.loads((FIXTURES / "tune-plan.json").read_text(encoding="utf-8"))


def make_plan(*lanes: dict) -> Plan:
    document = plan_document()
    template = document["lanes"][0]
    document["lanes"] = []
    document["identities"] = []
    for index, overrides in enumerate(lanes):
        lane = copy.deepcopy(template)
        lane["name"] = f"lane-{index}"
        envelope = overrides.pop("envelope", None)
        machine_profile = overrides.pop("machineProfile", None)
        lane.update(overrides)
        lane["envelope"] = envelope or {"floorMin": 0, "floorMax": 3}
        if machine_profile is not None:
            lane["cell"]["machineProfile"] = machine_profile
        document["lanes"].append(lane)
        warmth = "lazy-resident" if lane["envelope"]["floorMin"] >= 1 else "cold-with-bound"
        document["identities"].append({"identity": f"model-{index}", "lane": lane["name"], "warmth": warmth})
    return Plan.from_dict(document)


def one_lane(floor_min: int = 0, floor_max: int = 3) -> tuple[keda.Carrier, ...]:
    return keda.carriers(make_plan({"envelope": {"floorMin": floor_min, "floorMax": floor_max}}))


def lane_labels(queue_pool: str = "default", machine_profile: str = "l4", bundle: str = "default") -> dict[str, str]:
    return {"pool": queue_pool, "machine_profile": machine_profile, "bundle": bundle}


class FakePrometheus:
    def __init__(self, series: Mapping[str, Sequence[tuple[dict[str, str], float]]] | None = None) -> None:
        self.series = dict(series or {})
        self.failing: set[str] = set()
        self.queries: list[str] = []

    @classmethod
    def lane(
        cls,
        *,
        queue_depth: int = 0,
        pending: int = 0,
        warm_floor: int = 0,
        runners: int = 0,
        labels: dict | None = None,
    ) -> FakePrometheus:
        labels = labels or lane_labels()
        otel = f"{labels['pool']}|{labels['machine_profile']}|{labels['bundle']}"
        series: dict[str, list[tuple[dict[str, str], float]]] = {
            QUEUE_METRIC: [(labels, queue_depth)],
            PENDING_METRIC: [(labels, pending)],
            FLOOR_METRIC: [(labels, warm_floor)],
            SLOTS_METRIC: [({"lane": otel}, runners)] if runners else [],
        }
        return cls(series)

    def query(self, promql: str) -> Sequence[keda.Sample]:
        self.queries.append(promql)
        for metric, samples in self.series.items():
            if metric in promql:
                if metric in self.failing:
                    raise keda.PrometheusError("the query failed with HTTP 503")
                return tuple(keda.Sample(labels=labels, value=value) for labels, value in samples)
        for metric in self.failing:
            if metric in promql:
                raise keda.PrometheusError("the query failed with HTTP 503")
        return ()


class FakePools:
    def __init__(self, pools: Mapping[str, dict] | None = None) -> None:
        self.pools: dict[str, dict] = {name: copy.deepcopy(spec) for name, spec in (pools or {}).items()}
        self.raw: dict[str, dict] = {}
        self.reads: list[str] = []
        self.writes: list[tuple[tuple, dict]] = []
        self.fail_reads = False
        self.fail_writes = False

    def get_pool(self, name: str) -> Mapping[str, object] | None:
        self.reads.append(name)
        if self.fail_reads:
            raise keda.GatewayError("connection refused")
        if name in self.raw:
            return self.raw[name]
        spec = self.pools.get(name)
        return None if spec is None else {"name": name, "spec": copy.deepcopy(spec), "status": {"state": "active"}}

    def create_pool(
        self, name: str, gpus: dict[str, int], *, bundle: str, minimum_worker_count: int, queue_pool: str
    ) -> None:
        self.writes.append(
            ((name, gpus), {"bundle": bundle, "minimum_worker_count": minimum_worker_count, "queue_pool": queue_pool})
        )
        if self.fail_writes:
            raise keda.GatewayError("HTTP 403")
        self.pools[name] = {
            "name": name,
            "queue_pool": queue_pool,
            "bundle": bundle,
            "gpus": dict(gpus),
            "ttl_seconds": None,
            "minimum_worker_count": minimum_worker_count,
        }


def carrier_spec(floor: int, *, machine_profile: str = "l4", bundle: str = "default", **extra: object) -> dict:
    return {
        "name": f"placement-floor-default-{machine_profile}-{bundle}",
        "queue_pool": "default",
        "bundle": bundle,
        "gpus": {machine_profile: 0},
        "ttl_seconds": None,
        "minimum_worker_count": floor,
        **extra,
    }


def run_tick(
    lanes: Sequence[keda.Carrier],
    prometheus: FakePrometheus,
    reader: FakePools,
    writer: FakePools | None = None,
    selector: str = "",
) -> keda.Tick:
    return keda.tick(lanes, prometheus=prometheus, reader=reader, writer=writer, selector=selector, now=lambda: AT)


def test_each_fixture_lane_gets_one_floor_carrier():
    lanes = keda.carriers(Plan.from_dict(plan_document()))
    assert [carrier.name for carrier in lanes] == [
        "placement-floor-default-l4-default",
        "placement-floor-default-a100-80gb-generation",
    ]
    assert [carrier.gpus for carrier in lanes] == [{"l4": 0}, {"a100-80gb": 0}]
    assert lanes[0].key.otel_lane == "default|l4|default"


def test_lane_tokens_are_canonicalized_the_way_the_chart_names_lanes():
    lanes = keda.carriers(make_plan({"queuePool": "Default", "machineProfile": "L4-Spot", "bundle": "Default"}))
    assert lanes[0].key == keda.LaneKey(queue_pool="default", machine_profile="l4-spot", bundle="default")
    assert lanes[0].name == "placement-floor-default-l4-spot-default"


def test_an_envelope_above_the_gateway_per_pool_floor_limit_is_refused():
    plan = make_plan({"envelope": {"floorMin": 0, "floorMax": 5}, "maxContainers": 6})
    with pytest.raises(keda.PlanRefusedError) as raised:
        keda.carriers(plan)
    assert raised.value.problems == ("$.lanes[0].envelope.floorMax: 5 exceeds the gateway's per-pool floor limit of 4",)
    assert keda.carriers(plan, max_pool_floor=5)[0].lane.envelope.floor_max == 5


def test_more_lanes_than_the_gateway_pool_limit_are_refused():
    plan = make_plan({"bundle": "a"}, {"bundle": "b"}, {"bundle": "c"})
    with pytest.raises(keda.PlanRefusedError, match="3 lanes need as many floor-carrier pools"):
        keda.carriers(plan, max_pools=2)
    assert len(keda.carriers(plan, max_pools=3)) == 3


def test_two_lanes_that_need_one_carrier_are_refused():
    plan = make_plan({"bundle": "default"}, {"bundle": "DEFAULT"})
    with pytest.raises(keda.PlanRefusedError, match=r"\$\.lanes\[1\]: needs the same floor-carrier pool"):
        keda.carriers(plan)
    hyphens = make_plan({"queuePool": "a-b", "machineProfile": "c"}, {"queuePool": "a", "machineProfile": "b-c"})
    with pytest.raises(keda.PlanRefusedError, match="placement-floor-a-b-c-default"):
        keda.carriers(hyphens)


@pytest.mark.parametrize("token", ["l4.x", "l4/x", "x" * 64, 'l4"'])
def test_a_lane_token_the_gateway_would_reject_is_refused(token):
    with pytest.raises(keda.PlanRefusedError, match="must each be 1 to 63 letters"):
        keda.carriers(make_plan({"machineProfile": token}))


def test_a_carrier_name_longer_than_the_gateway_accepts_is_refused():
    plan = make_plan({"queuePool": "q" * 40, "machineProfile": "m" * 40, "bundle": "b" * 40})
    with pytest.raises(keda.PlanRefusedError, match="longer than 128 characters"):
        keda.carriers(plan)


def test_a_pin_on_the_default_queue_pool_is_refused_as_the_helm_emitter_refuses_it():
    document = plan_document()
    document["lanes"][0]["queuePool"] = "Default"
    document["identities"][1]["warmth"] = "pinned"
    document["identities"][0]["warmth"] = "pinned"
    with pytest.raises(keda.PlanRefusedError) as raised:
        keda.carriers(Plan.from_dict(document))
    assert raised.value.problems == (
        "$.identities[0].warmth: pins 'BAAI/bge-m3' on chart lane 'default/l4/default', but the gateway keeps no "
        "static queue pool for the 'default' queue pool, so its pinnedModels have no carrier; place pinned "
        "identities on a named queuePool",
    )
    document["lanes"][0]["queuePool"] = "embed"
    (pinned, _) = keda.carriers(Plan.from_dict(document))
    assert pinned.name == "placement-floor-embed-l4-default"


@pytest.mark.parametrize("warmth", sorted(keda.UNREALIZED_WARMTHS))
def test_a_warmth_the_chart_does_not_realize_is_refused_as_the_helm_emitter_refuses_it(warmth):
    document = plan_document()
    document["identities"][2]["warmth"] = warmth.value
    with pytest.raises(keda.PlanRefusedError) as raised:
        keda.carriers(Plan.from_dict(document))
    assert raised.value.problems == (
        f"$.identities[2].warmth: is {warmth.value!r} for 'Qwen/Qwen3-8B' on lane 'default-a100-80gb-generation', "
        f"which the sie-cluster chart does not realize: {keda.UNREALIZED_WARMTHS[warmth]}; plan the identity "
        "pinned, lazy-resident or cold-with-bound",
    )


def test_every_problem_of_a_refused_plan_is_reported():
    document = plan_document()
    document["identities"][0]["warmth"] = "pinned"
    document["identities"][2]["warmth"] = "snapshot-cold"
    document["lanes"][1]["envelope"]["floorMax"] = 5
    document["lanes"][1]["maxContainers"] = 6
    with pytest.raises(keda.PlanRefusedError) as raised:
        keda.carriers(Plan.from_dict(document))
    assert [problem.split(":", 1)[0] for problem in raised.value.problems] == [
        "$.identities[0].warmth",
        "$.lanes[1].envelope.floorMax",
        "$.identities[2].warmth",
    ]


def test_the_series_queries_read_each_lane_from_fresh_gateway_producers_and_the_worker_series():
    capacity_fresh = (
        "and on(producer_instance, collector_generation) "
        "(abs(time() - sie_gateway_capacity_snapshot_timestamp_seconds{}) < 20))"
    )
    assert keda.series_queries() == {
        "queueDepth": (
            "max by (pool, machine_profile, bundle) (sie_gateway_lane_queue_depth{} "
            "and on(producer_instance, collector_generation, pool, machine_profile, bundle) "
            "(abs(time() - sie_gateway_lane_queue_snapshot_timestamp_seconds{}) < 20))"
        ),
        "pendingDemand": f"max by (pool, machine_profile, bundle) (sie_gateway_pending_demand{{}} {capacity_fresh}",
        "warmFloor": f"max by (pool, machine_profile, bundle) (sie_gateway_pool_warm_floor{{}} {capacity_fresh}",
        "runners": ('count by (lane) (max by (lane, producer_instance) (sie_worker_gpu_slots{state="ready"}) > 0)'),
    }
    scoped = keda.series_queries('namespace="sie"')
    assert scoped["queueDepth"] == (
        'max by (pool, machine_profile, bundle) (sie_gateway_lane_queue_depth{namespace="sie"} '
        "and on(producer_instance, collector_generation, pool, machine_profile, bundle) "
        '(abs(time() - sie_gateway_lane_queue_snapshot_timestamp_seconds{namespace="sie"}) < 20))'
    )
    for name in ("pendingDemand", "warmFloor"):
        assert scoped[name].endswith(
            '{namespace="sie"} and on(producer_instance, collector_generation) '
            '(abs(time() - sie_gateway_capacity_snapshot_timestamp_seconds{namespace="sie"}) < 20))'
        )
    assert scoped["runners"].endswith('(sie_worker_gpu_slots{namespace="sie",state="ready"}) > 0)')


def test_a_lane_whose_gateway_producers_are_all_stale_is_held():
    stale = lane_labels(bundle="stale")
    prometheus = FakePrometheus.lane(queue_depth=4, pending=1, runners=2)
    prometheus.series[QUEUE_METRIC] = [(lane_labels(), 4)]
    prometheus.series[PENDING_METRIC] = [(lane_labels(), 1)]
    lanes = keda.carriers(make_plan({"bundle": "default"}, {"bundle": "stale"}))
    pools = FakePools({carrier.name: carrier_spec(1, bundle=carrier.key.bundle) for carrier in lanes})
    fresh, held = run_tick(lanes, prometheus, pools, pools).records
    assert held.carrier.key.bundle == stale["bundle"]
    assert (fresh.step.action, fresh.step.settings.floor) == (Action.SCALE_UP, 2)
    assert (held.telemetry.queue_depth, held.telemetry.pending_demand) == (None, None)
    assert (held.step.action, held.step.settings.floor) == (Action.HOLD, 1)


@pytest.mark.parametrize("selector", ['namespace="sie"}', '{namespace="sie"', 'a="b"\nc="d"'])
def test_a_selector_that_could_leave_its_braces_is_rejected(selector):
    with pytest.raises(ValueError, match="without braces or line breaks"):
        keda.series_queries(selector)


def test_parse_vector_reads_an_instant_vector():
    document = {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {"pool": "default"}, "value": [1.0, "3"]}]},
    }
    assert keda.parse_vector(document) == (keda.Sample(labels={"pool": "default"}, value=3.0),)


@pytest.mark.parametrize(
    "document",
    [
        {"status": "error", "error": "bad query"},
        {"status": "success", "data": {"resultType": "matrix", "result": []}},
        {"status": "success", "data": {"resultType": "vector", "result": {}}},
        {"status": "success", "data": {"resultType": "vector", "result": [{"metric": {}}]}},
        {"status": "success", "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1, "x"]}]}},
        [],
    ],
)
def test_parse_vector_rejects_anything_else(document):
    with pytest.raises(keda.PrometheusError):
        keda.parse_vector(document)


class _PrometheusHandler(BaseHTTPRequestHandler):
    server: _PrometheusServer

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        self.server.requests.append((url.path, parse_qs(url.query)))
        status, body = self.server.reply
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class _PrometheusServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _PrometheusHandler)
        self.requests: list[tuple[str, dict]] = []
        self.reply: tuple[int, object] = (200, {"status": "success", "data": {"resultType": "vector", "result": []}})

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def prometheus_server(monkeypatch) -> Iterator[_PrometheusServer]:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    server = _PrometheusServer()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_the_http_reader_sends_the_query_to_the_query_api(prometheus_server):
    prometheus_server.reply = (
        200,
        {"status": "success", "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1, "2"]}]}},
    )
    reader = keda.PrometheusHttp(prometheus_server.url + "/")
    assert reader.query('max(up{job="x"})') == (keda.Sample(labels={}, value=2.0),)
    assert prometheus_server.requests == [("/api/v1/query", {"query": ['max(up{job="x"})']})]


def test_the_http_reader_reports_a_failed_query(prometheus_server):
    prometheus_server.reply = (400, {"status": "error", "error": "parse error"})
    with pytest.raises(keda.PrometheusError, match="HTTP 400"):
        keda.PrometheusHttp(prometheus_server.url).query("up{")


def test_the_http_reader_reports_an_unreachable_prometheus():
    with pytest.raises(keda.PrometheusError, match="the query failed"):
        keda.PrometheusHttp("http://127.0.0.1:9", timeout_s=2).query("up")


class _RawHandler(socketserver.StreamRequestHandler):
    server: _RawServer

    def handle(self) -> None:
        self.rfile.readline()
        self.wfile.write(self.server.reply)


class _RawServer(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, reply: bytes) -> None:
        super().__init__(("127.0.0.1", 0), _RawHandler)
        self.reply = reply


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (b"garbage\r\n\r\n", "BadStatusLine"),
        (
            b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{"status": "s',
            "IncompleteRead",
        ),
    ],
)
def test_the_http_reader_reports_a_malformed_reply(monkeypatch, reply, error):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    server = _RawServer(reply)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        with pytest.raises(keda.PrometheusError, match=f"the query failed: {error}"):
            keda.PrometheusHttp(f"http://127.0.0.1:{server.server_address[1]}", timeout_s=2).query("up")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://prometheus",
        "prometheus:9090",
        "http://user:secret@prometheus",
        "http://p/?x=1",
        "http://prometheus:notaport",
        "http://prometheus:99999",
    ],
)
def test_the_http_reader_accepts_only_a_plain_http_base_url(url):
    with pytest.raises(ValueError, match="Prometheus URL"):
        keda.PrometheusHttp(url)


def test_shadow_mode_decides_every_lane_and_writes_nothing():
    lanes = keda.carriers(Plan.from_dict(plan_document()))
    prometheus = FakePrometheus.lane(queue_depth=4, pending=1, warm_floor=1, runners=1)
    reader = FakePools({"placement-floor-default-l4-default": carrier_spec(1)})
    result = run_tick(lanes, prometheus, reader)
    assert result.apply is False
    assert reader.writes == []
    assert reader.reads == [carrier.name for carrier in lanes]
    first, second = result.records
    assert (first.current_floor, first.step.action, first.step.settings.floor, first.applied) == (
        1,
        Action.SCALE_UP,
        2,
        False,
    )
    assert (second.current_floor, second.step.action, second.step.settings.floor) == (0, Action.HOLD, 0)
    assert second.telemetry == keda.LaneTelemetry(queue_depth=None, pending_demand=None, warm_floor=None, runners=0)


def test_apply_upserts_one_unpinned_carrier_per_lane_through_the_pool_api():
    lanes = one_lane(1, 3)
    pools = FakePools()
    result = run_tick(lanes, FakePrometheus.lane(), pools, pools)
    assert pools.writes == [
        (
            ("placement-floor-default-l4-default", {"l4": 0}),
            {"bundle": "default", "minimum_worker_count": 1, "queue_pool": "default"},
        )
    ]
    assert result.records[0].step.action is Action.RAISE_TO_FLOOR
    assert result.records[0].applied is True


def test_apply_renews_the_carrier_lease_on_every_tick():
    lanes = one_lane(1, 3)
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(2)})
    busy = FakePrometheus.lane(queue_depth=3, runners=2)
    for _ in range(3):
        result = run_tick(lanes, busy, pools, pools)
        assert result.records[0].step.action is Action.HOLD
    assert [kwargs["minimum_worker_count"] for _, kwargs in pools.writes] == [2, 2, 2]


def test_a_lane_without_a_carrier_that_stays_at_zero_gets_no_carrier():
    pools = FakePools()
    result = run_tick(one_lane(0, 3), FakePrometheus.lane(), pools, pools)
    assert pools.writes == []
    assert result.records[0].step.action is Action.HOLD
    assert result.records[0].applied is False


def test_an_existing_carrier_at_zero_is_still_renewed():
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(0)})
    run_tick(one_lane(0, 3), FakePrometheus.lane(), pools, pools)
    assert [kwargs["minimum_worker_count"] for _, kwargs in pools.writes] == [0]


@pytest.mark.parametrize(("floor_min", "floor_max"), [(0, 0), (0, 3), (1, 1), (1, 4), (2, 3)])
def test_the_written_floor_never_leaves_the_envelope(floor_min, floor_max):
    lanes = one_lane(floor_min, floor_max)
    for current, queue_depth, pending, runners in itertools.product(range(7), (0, 3), (0, 1), (0, 1, 3, 6)):
        pools = FakePools({"placement-floor-default-l4-default": carrier_spec(current)})
        prometheus = FakePrometheus.lane(queue_depth=queue_depth, pending=pending, runners=runners)
        record = run_tick(lanes, prometheus, pools, pools).records[0]
        (_, kwargs) = pools.writes[-1]
        assert floor_min <= kwargs["minimum_worker_count"] <= floor_max
        assert kwargs["minimum_worker_count"] == record.step.settings.floor
        assert record.current_floor == current


def test_pending_demand_raises_the_floor_and_an_idle_lane_lowers_it():
    lanes = one_lane(0, 3)
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(1)})
    waiting = FakePrometheus.lane(queue_depth=2, pending=1, runners=1)
    serving = FakePrometheus.lane(queue_depth=2, pending=0, runners=1)
    idle = FakePrometheus.lane(queue_depth=0, pending=0, runners=1)
    assert run_tick(lanes, waiting, pools, pools).records[0].step.action is Action.SCALE_UP
    assert pools.pools["placement-floor-default-l4-default"]["minimum_worker_count"] == 2
    assert run_tick(lanes, serving, pools, pools).records[0].step.action is Action.HOLD
    assert run_tick(lanes, idle, pools, pools).records[0].step.action is Action.SCALE_DOWN
    assert pools.pools["placement-floor-default-l4-default"]["minimum_worker_count"] == 1


def test_a_floor_whose_workers_have_not_started_is_not_raised_again():
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(2)})
    booting = FakePrometheus.lane(queue_depth=5, pending=1, runners=1)
    record = run_tick(one_lane(0, 3), booting, pools, pools).records[0]
    assert (record.step.action, record.step.settings.floor) == (Action.HOLD, 2)


def test_current_is_read_from_the_gateway_on_every_tick():
    lanes = one_lane(0, 3)
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(1)})
    waiting = FakePrometheus.lane(queue_depth=2, pending=1, runners=3)
    assert run_tick(lanes, waiting, pools, pools).records[0].step.settings.floor == 2
    pools.pools["placement-floor-default-l4-default"]["minimum_worker_count"] = 0
    record = run_tick(lanes, waiting, pools, pools).records[0]
    assert (record.current_floor, record.step.settings.floor) == (0, 1)


def test_a_floor_held_above_the_envelope_by_another_pool_is_reported_and_left_alone():
    lanes = one_lane(1, 3)
    other = {
        "name": "team-pool",
        "queue_pool": "default",
        "bundle": "default",
        "gpus": {"l4": 1},
        "minimum_worker_count": 4,
    }
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(1), "team-pool": other})
    record = run_tick(lanes, FakePrometheus.lane(warm_floor=4, runners=4), pools, pools).records[0]
    assert record.lane_floor_above_envelope is True
    assert record.to_dict()["laneFloorAboveEnvelope"] is True
    assert [args[0] for args, _ in pools.writes] == ["placement-floor-default-l4-default"]
    assert pools.pools["team-pool"] == other
    assert (
        run_tick(lanes, FakePrometheus.lane(warm_floor=3), pools, pools).records[0].lane_floor_above_envelope is False
    )


def test_a_carrier_above_a_lowered_envelope_is_brought_down_to_the_cap():
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(4)})
    record = run_tick(one_lane(0, 2), FakePrometheus.lane(warm_floor=4, runners=4), pools, pools).records[0]
    assert (record.step.action, record.step.settings.floor) == (Action.HOLD_AT_CAP, 2)
    assert pools.pools["placement-floor-default-l4-default"]["minimum_worker_count"] == 2


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"pinned_models": ["BAAI/bge-m3"]}, "it sets pinned_models"),
        ({"gpu_caps": {"l4": 2}}, "it sets gpu_caps"),
        ({"ttl_seconds": 3600}, "it sets ttl_seconds"),
        ({"gpus": {"l4": 1}}, "its gpus are not"),
        ({"gpus": {"l4": 0, "a100": 0}}, "its gpus are not"),
        ({"bundle": "other"}, "bundle 'default'"),
        ({"bundle": None}, "bundle 'default'"),
        ({"queue_pool": "dedicated"}, "queue pool 'default'"),
        ({"minimum_worker_count": -1}, "minimum_worker_count"),
        ({"minimum_worker_count": True}, "minimum_worker_count"),
    ],
)
def test_a_pool_that_is_not_the_lane_carrier_is_never_written(change, reason):
    spec = carrier_spec(2) | change
    pools = FakePools({"placement-floor-default-l4-default": spec})
    record = run_tick(one_lane(0, 3), FakePrometheus.lane(pending=1, runners=2), pools, pools).records[0]
    assert pools.writes == []
    assert record.step is None
    assert reason in (record.error or "")
    assert "is not this lane's floor carrier" in (record.error or "")


def test_a_pool_without_a_spec_is_never_written():
    pools = FakePools()
    pools.raw["placement-floor-default-l4-default"] = {"name": "placement-floor-default-l4-default"}
    record = run_tick(one_lane(0, 3), FakePrometheus.lane(), pools, pools).records[0]
    assert record.error is not None
    assert pools.writes == []


def test_missing_telemetry_holds_the_lane_and_only_brings_it_inside_the_envelope():
    lanes = one_lane(1, 3)
    prometheus = FakePrometheus.lane(pending=1, runners=3)
    prometheus.failing.add(QUEUE_METRIC)
    for current, expected, action in ((0, 1, Action.RAISE_TO_FLOOR), (2, 2, Action.HOLD), (5, 3, Action.HOLD_AT_CAP)):
        pools = FakePools({"placement-floor-default-l4-default": carrier_spec(current)})
        result = run_tick(lanes, prometheus, pools, pools)
        record = result.records[0]
        assert (record.step.action, record.step.settings.floor) == (action, expected)
        assert record.telemetry.queue_depth is None
        assert result.series_errors == ("queueDepth: the query failed with HTTP 503",)
        assert result.failed is True


def test_a_lane_the_gateway_does_not_report_is_held():
    prometheus = FakePrometheus.lane(pending=1, runners=1, labels=lane_labels(bundle="other"))
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(1)})
    result = run_tick(one_lane(0, 3), prometheus, pools, pools)
    record = result.records[0]
    assert (record.step.action, record.step.settings.floor) == (Action.HOLD, 1)
    assert result.failed is False


def test_a_lane_with_no_ready_worker_reports_zero_runners():
    prometheus = FakePrometheus.lane(pending=1)
    record = run_tick(one_lane(0, 3), prometheus, FakePools(), FakePools()).records[0]
    assert record.telemetry.runners == 0
    assert (record.step.action, record.step.settings.floor) == (Action.SCALE_UP, 1)


def test_samples_are_matched_to_lanes_by_canonical_labels():
    labels = {"pool": "Default", "machine_profile": "L4", "bundle": "Default"}
    prometheus = FakePrometheus(
        {
            QUEUE_METRIC: [(labels, 2), (lane_labels(), 5), ({"pool": "default"}, 9)],
            PENDING_METRIC: [(labels, 0), (lane_labels(), float("nan"))],
            FLOOR_METRIC: [(labels, -1)],
            SLOTS_METRIC: [({"lane": "Default|L4|Default"}, 2), ({"lane": "default|l4"}, 7)],
        }
    )
    record = run_tick(one_lane(0, 3), prometheus, FakePools()).records[0]
    assert record.telemetry == keda.LaneTelemetry(queue_depth=5, pending_demand=0, warm_floor=None, runners=2)


def test_a_gateway_read_failure_skips_only_that_lane():
    lanes = keda.carriers(make_plan({"bundle": "a"}, {"bundle": "b"}))
    reader = FakePools()
    reader.fail_reads = True
    writer = FakePools()
    result = run_tick(lanes, FakePrometheus.lane(), reader, writer)
    assert writer.writes == []
    assert [record.error for record in result.records] == [
        "reading pool 'placement-floor-default-l4-a' failed: connection refused",
        "reading pool 'placement-floor-default-l4-b' failed: connection refused",
    ]
    assert result.failed is True


def test_a_gateway_write_failure_is_recorded():
    pools = FakePools()
    pools.fail_writes = True
    record = run_tick(one_lane(1, 3), FakePrometheus.lane(), pools, pools).records[0]
    assert record.applied is False
    assert record.step.settings.floor == 1
    assert record.error == "writing pool 'placement-floor-default-l4-default' failed: HTTP 403"


def test_audit_rows_carry_the_tick_time_mode_and_decision():
    pools = FakePools({"placement-floor-default-l4-default": carrier_spec(1)})
    prometheus = FakePrometheus.lane(queue_depth=4, pending=1, warm_floor=1, runners=1)
    rows = run_tick(one_lane(1, 3), prometheus, pools, pools).audit_rows()
    assert rows == [
        {
            "at": "2026-01-01T12:00:00Z",
            "mode": "apply",
            "lane": "lane-0",
            "queuePool": "default",
            "machineProfile": "l4",
            "bundle": "default",
            "carrier": "placement-floor-default-l4-default",
            "telemetry": {"queueDepth": 4, "pendingDemand": 1, "warmFloor": 1, "runners": 1},
            "currentFloor": 1,
            "action": "scale_up",
            "floor": 2,
            "applied": True,
            "laneFloorAboveEnvelope": False,
            "error": None,
        }
    ]
    assert json.loads(json.dumps(rows)) == rows


def test_the_selector_scopes_every_series_query():
    prometheus = FakePrometheus.lane()
    run_tick(one_lane(), prometheus, FakePools(), selector='namespace="sie"')
    assert len(prometheus.queries) == 4
    assert all('namespace="sie"' in query for query in prometheus.queries)
