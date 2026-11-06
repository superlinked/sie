from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("sie_sdk")

from sie_placement import cli, keda, pools

FIXTURES = Path(__file__).parent / "fixtures"
READ_TOKEN = "read-token-for-tests"  # noqa: S105 - a test value
ADMIN_TOKEN = "admin-token-for-tests"  # noqa: S105 - a test value
L4 = {"pool": "default", "machine_profile": "l4", "bundle": "default"}
A100 = {"pool": "default", "machine_profile": "a100-80gb", "bundle": "generation"}


def vector(*samples: tuple[dict[str, str], float]) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": labels, "value": [0, str(value)]} for labels, value in samples],
        },
    }


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def _reply(self, status: int, body: object) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> object:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else None

    def do_GET(self) -> None:
        self.server.respond(self, "GET", None)

    def do_POST(self) -> None:
        self.server.respond(self, "POST", self._body())

    def log_message(self, format: str, *args: object) -> None:
        return


class _Server(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def respond(self, handler: _Handler, method: str, body: object) -> None:
        raise NotImplementedError


class FakeGateway(_Server):
    def __init__(self, *, auth: bool = True) -> None:
        super().__init__()
        self.auth = auth
        self.pools: dict[str, dict] = {}
        self.fail_reads = False
        self.read_reply: object | None = None
        self.raw_reply: tuple[int, bytes] | None = None

    def respond(self, handler: _Handler, method: str, body: object) -> None:
        path = urlsplit(handler.path).path
        authorization = handler.headers.get("Authorization")
        self.requests.append({"method": method, "path": path, "authorization": authorization, "body": body})
        if self.raw_reply is not None:
            status, payload = self.raw_reply
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.send_header("Location", "http://127.0.0.1:9/v1/pools")
            handler.end_headers()
            handler.wfile.write(payload)
            return
        if method == "GET" and path.startswith("/v1/pools/"):
            if self.read_reply is not None:
                handler._reply(200, self.read_reply)
                return
            if self.fail_reads or (self.auth and authorization != f"Bearer {READ_TOKEN}"):
                handler._reply(401, {"detail": {"code": "UNAUTHORIZED", "message": "Invalid token"}})
                return
            spec = self.pools.get(path.removeprefix("/v1/pools/"))
            if spec is None:
                handler._reply(404, {"detail": {"code": "POOL_NOT_FOUND", "message": "Pool not found"}})
                return
            handler._reply(200, {"spec": spec, "status": {"state": "active", "assigned_workers": []}})
            return
        if method == "POST" and path.startswith("/v1/pools"):
            if self.auth and authorization != f"Bearer {ADMIN_TOKEN}":
                handler._reply(403, {"detail": {"code": "FORBIDDEN", "message": "Admin token required"}})
                return
            if path == "/v1/pools" and isinstance(body, dict):
                spec = {
                    "name": body["name"],
                    "queue_pool": body.get("queue_pool", "default"),
                    "bundle": body.get("bundle"),
                    "gpus": body.get("gpus", {}),
                    "ttl_seconds": body.get("ttl_seconds"),
                    "minimum_worker_count": body.get("minimum_worker_count", 0),
                }
                self.pools[body["name"]] = spec
                handler._reply(201, {"spec": spec, "status": {"state": "pending", "assigned_workers": []}})
                return
            handler._reply(200, {"message": "Pool renewed"})
            return
        handler._reply(404, {"detail": {"code": "NOT_FOUND", "message": "not found"}})


class FakePrometheus(_Server):
    def __init__(self) -> None:
        super().__init__()
        self.series: dict[str, dict] = {
            "sie_gateway_lane_queue_depth": vector((L4, 3), (A100, 0)),
            "sie_gateway_pending_demand": vector((L4, 1), (A100, 0)),
            "sie_gateway_pool_warm_floor": vector((L4, 1), (A100, 0)),
            "sie_worker_gpu_slots": vector(({"lane": "default|l4|default"}, 1)),
        }
        self.truncate = False

    def respond(self, handler: _Handler, method: str, body: object) -> None:
        url = urlsplit(handler.path)
        query = parse_qs(url.query).get("query", [""])[0]
        self.requests.append({"path": url.path, "query": query, "authorization": handler.headers.get("Authorization")})
        if self.truncate:
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", "500")
            handler.end_headers()
            handler.wfile.write(b'{"status": "succ')
            return
        for metric, document in self.series.items():
            if metric in query:
                handler._reply(200, document)
                return
        handler._reply(200, vector())


def _serve(server: _Server) -> Iterator[_Server]:
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "SIE_API_KEY",
        "SIE_BASE_URL",
        "SIE_ADMIN_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gateway() -> Iterator[FakeGateway]:
    yield from _serve(FakeGateway())


@pytest.fixture
def prometheus() -> Iterator[FakePrometheus]:
    yield from _serve(FakePrometheus())


@pytest.fixture
def read_token_file(tmp_path) -> Path:
    path = tmp_path / "read-token"
    path.write_text(READ_TOKEN + "\n", encoding="utf-8")
    return path


def run(gateway: FakeGateway, prometheus: FakePrometheus, *extra: str, plan: Path = FIXTURES / "tune-plan.json") -> int:
    return cli.main(
        ["tune", "--plan", str(plan), "--prometheus-url", prometheus.url, "--gateway-url", gateway.url, *extra]
    )


def tune(
    gateway: FakeGateway, prometheus: FakePrometheus, *extra: str, plan: Path = FIXTURES / "tune-plan.json"
) -> int:
    return run(gateway, prometheus, "--once", *extra, plan=plan)


def rows(captured: str) -> list[dict]:
    return [json.loads(line) for line in captured.splitlines()]


def assert_no_secret(capsys_text: str, caplog) -> None:
    for secret in (READ_TOKEN, ADMIN_TOKEN):
        assert secret not in capsys_text
        assert secret not in caplog.text


def test_shadow_mode_records_decisions_and_writes_no_pool(gateway, prometheus, read_token_file, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file)) == 0
    captured = capsys.readouterr()
    lines = rows(captured.out)
    assert [(row["mode"], row["carrier"], row["action"], row["floor"], row["applied"]) for row in lines] == [
        ("shadow", "placement-floor-default-l4-default", "raise_to_floor", 1, False),
        ("shadow", "placement-floor-default-a100-80gb-generation", "hold", 0, False),
    ]
    assert lines[0]["telemetry"] == {"queueDepth": 3, "pendingDemand": 1, "warmFloor": 1, "runners": 1}
    assert [request["method"] for request in gateway.requests] == ["GET", "GET"]
    assert {request["authorization"] for request in gateway.requests} == {f"Bearer {READ_TOKEN}"}
    assert [request["path"] for request in prometheus.requests] == ["/api/v1/query"] * 4
    assert_no_secret(captured.out + captured.err, caplog)


def test_shadow_mode_never_reads_the_admin_token(gateway, prometheus, read_token_file, monkeypatch, capsys):
    monkeypatch.setenv("SIE_ADMIN_TOKEN", ADMIN_TOKEN)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--admin-token-file", "/nonexistent") == 0
    assert all(request["method"] == "GET" for request in gateway.requests)
    assert all(request["authorization"] != f"Bearer {ADMIN_TOKEN}" for request in gateway.requests)
    capsys.readouterr()


def test_apply_upserts_carriers_with_the_admin_token_only_on_writes(
    gateway, prometheus, read_token_file, monkeypatch, capsys, caplog
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("SIE_ADMIN_TOKEN", ADMIN_TOKEN)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply") == 0
    captured = capsys.readouterr()
    writes = [request for request in gateway.requests if request["method"] == "POST"]
    reads = [request for request in gateway.requests if request["method"] == "GET"]
    assert [request["body"] for request in writes] == [
        {
            "name": "placement-floor-default-l4-default",
            "gpus": {"l4": 0},
            "queue_pool": "default",
            "bundle": "default",
            "minimum_worker_count": 1,
        }
    ]
    assert {request["authorization"] for request in writes} == {f"Bearer {ADMIN_TOKEN}"}
    assert {request["authorization"] for request in reads} == {f"Bearer {READ_TOKEN}"}
    assert [(row["mode"], row["applied"]) for row in rows(captured.out)] == [("apply", True), ("apply", False)]
    assert_no_secret(captured.out + captured.err, caplog)


def test_apply_reads_the_admin_token_from_a_file(gateway, prometheus, read_token_file, tmp_path, capsys):
    admin = tmp_path / "admin-token"
    admin.write_text(ADMIN_TOKEN, encoding="utf-8")
    assert (
        tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply", "--admin-token-file", str(admin))
        == 0
    )
    writes = [request for request in gateway.requests if request["method"] == "POST"]
    assert {request["authorization"] for request in writes} == {f"Bearer {ADMIN_TOKEN}"}
    capsys.readouterr()


def test_apply_renews_an_existing_carrier_and_steers_from_its_floor(
    gateway, prometheus, read_token_file, monkeypatch, capsys
):
    monkeypatch.setenv("SIE_ADMIN_TOKEN", ADMIN_TOKEN)
    gateway.pools["placement-floor-default-l4-default"] = {
        "name": "placement-floor-default-l4-default",
        "queue_pool": "default",
        "bundle": "default",
        "gpus": {"l4": 0},
        "ttl_seconds": None,
        "minimum_worker_count": 1,
    }
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply") == 0
    l4_row = rows(capsys.readouterr().out)[0]
    assert (l4_row["currentFloor"], l4_row["action"], l4_row["floor"]) == (1, "scale_up", 2)
    assert gateway.pools["placement-floor-default-l4-default"]["minimum_worker_count"] == 2


def test_the_client_token_from_the_environment_is_never_sent_on_writes(prometheus, monkeypatch, capsys):
    unauthenticated = FakeGateway(auth=False)
    for server in _serve(unauthenticated):
        monkeypatch.setenv("SIE_BASE_URL", server.url)
        monkeypatch.setenv("SIE_API_KEY", READ_TOKEN)
        assert tune(server, prometheus, "--apply") == 0
        reads = [request for request in server.requests if request["method"] == "GET"]
        writes = [request for request in server.requests if request["method"] == "POST"]
        assert {request["authorization"] for request in reads} == {f"Bearer {READ_TOKEN}"}
        assert writes
        assert {request["authorization"] for request in writes} == {None}
    assert "pool writes carry no credential" in capsys.readouterr().err


def test_a_rejected_write_fails_the_tick_and_names_the_pool(gateway, prometheus, read_token_file, capsys):
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply") == 1
    l4_row = rows(capsys.readouterr().out)[0]
    assert l4_row["applied"] is False
    assert l4_row["error"].startswith("writing pool 'placement-floor-default-l4-default' failed")
    assert "Admin token required" in l4_row["error"]


def test_an_envelope_above_the_pool_floor_limit_is_refused_before_any_call(
    gateway, prometheus, read_token_file, tmp_path, capsys
):
    document = json.loads((FIXTURES / "tune-plan.json").read_text(encoding="utf-8"))
    document["lanes"][0]["envelope"]["floorMax"] = 5
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    assert tune(gateway, prometheus, "--apply", plan=plan) == cli.EXIT_REFUSED
    assert "floorMax: 5 exceeds the gateway's per-pool floor limit of 4" in capsys.readouterr().err
    assert gateway.requests == []
    assert prometheus.requests == []
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--max-pool-floor", "5", plan=plan) == 0
    capsys.readouterr()


def test_a_plan_the_chart_cannot_run_is_refused_before_any_call(gateway, prometheus, tmp_path, capsys):
    document = json.loads((FIXTURES / "tune-plan.json").read_text(encoding="utf-8"))
    document["identities"][0]["warmth"] = "pinned"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    assert tune(gateway, prometheus, plan=plan) == cli.EXIT_REFUSED
    assert "pins 'BAAI/bge-m3' on chart lane 'default/l4/default'" in capsys.readouterr().err
    assert gateway.requests == []
    assert prometheus.requests == []


def test_more_lanes_than_the_pool_limit_are_refused_before_any_call(gateway, prometheus, capsys):
    assert tune(gateway, prometheus, "--apply", "--max-pools", "1") == cli.EXIT_REFUSED
    assert "above the gateway's limit of 1 API pools" in capsys.readouterr().err
    assert gateway.requests == []
    assert prometheus.requests == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--api-key-file", "/nonexistent/token"), "cannot read the gateway client token file"),
        (("--apply", "--admin-token-file", "/nonexistent/token"), "cannot read the gateway admin token file"),
        (("--series-selector", 'a="b"}'), "without braces"),
    ],
)
def test_bad_inputs_are_refused_before_any_call(gateway, prometheus, extra, message, capsys):
    assert tune(gateway, prometheus, *extra) == cli.EXIT_REFUSED
    assert message in capsys.readouterr().err
    assert gateway.requests == []


def test_an_empty_token_file_is_refused(gateway, prometheus, tmp_path, capsys):
    empty = tmp_path / "token"
    empty.write_text("\n", encoding="utf-8")
    assert tune(gateway, prometheus, "--api-key-file", str(empty)) == cli.EXIT_REFUSED
    assert "is empty" in capsys.readouterr().err


def test_a_plan_that_is_not_a_plan_is_refused(gateway, prometheus, tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    assert tune(gateway, prometheus, plan=plan) == cli.EXIT_REFUSED
    assert "$.kind: is required" in capsys.readouterr().err
    plan.write_text("{", encoding="utf-8")
    assert tune(gateway, prometheus, plan=plan) == cli.EXIT_REFUSED
    assert "is not JSON" in capsys.readouterr().err


def test_an_unreachable_prometheus_holds_every_lane_and_fails_the_tick(gateway, read_token_file, capsys):
    unreachable = FakePrometheus()
    unreachable.server_close()
    assert tune(gateway, unreachable, "--api-key-file", str(read_token_file)) == 1
    captured = capsys.readouterr()
    assert [row["action"] for row in rows(captured.out)] == ["raise_to_floor", "hold"]
    assert "Prometheus series queueDepth: the query failed" in captured.err


def test_the_core_imports_without_the_sdk_and_tune_names_the_missing_extra(gateway, prometheus, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sie_sdk", None)
    monkeypatch.delitem(sys.modules, "sie_placement.pools", raising=False)
    monkeypatch.delattr(sys.modules["sie_placement"], "pools", raising=False)
    assert tune(gateway, prometheus) == cli.EXIT_REFUSED
    assert "needs the sie-placement keda extra" in capsys.readouterr().err
    assert gateway.requests == []


TWO_LINE_SECRET = "first-line-of-a-secret\nsecond-line-of-a-secret"  # noqa: S105 - a test value


def assert_secret_absent(secret: str, captured, caplog) -> None:
    for part in (secret, *secret.split("\n")):
        assert part not in captured.out
        assert part not in captured.err
        assert part not in caplog.text


@pytest.mark.parametrize("secret", [TWO_LINE_SECRET, "secret\x00value"])
@pytest.mark.parametrize("flag", ["--api-key-file", "--admin-token-file"])
def test_a_token_file_with_a_control_character_is_refused_without_echoing_it(
    gateway, prometheus, tmp_path, flag, secret, capsys, caplog
):
    caplog.set_level(logging.DEBUG)
    token = tmp_path / "token"
    token.write_text(secret + "\n", encoding="utf-8")
    assert tune(gateway, prometheus, "--apply", flag, str(token)) == cli.EXIT_REFUSED
    captured = capsys.readouterr()
    assert "must be one run of printable ASCII characters without spaces" in captured.err
    assert_secret_absent(secret, captured, caplog)
    assert gateway.requests == []
    assert prometheus.requests == []


@pytest.mark.parametrize("value", [TWO_LINE_SECRET, "secret with spaces", "caf\u00e9-secret"])
def test_an_admin_token_variable_that_a_header_cannot_carry_is_refused(
    gateway, prometheus, read_token_file, monkeypatch, value, capsys, caplog
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("SIE_ADMIN_TOKEN", value)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply") == cli.EXIT_REFUSED
    captured = capsys.readouterr()
    assert "the gateway admin token in $SIE_ADMIN_TOKEN must be one run of printable ASCII" in captured.err
    assert_secret_absent(value, captured, caplog)
    assert gateway.requests == []


def test_a_token_file_that_is_not_utf8_is_refused_without_echoing_it(gateway, prometheus, tmp_path, capsys):
    token = tmp_path / "token"
    token.write_bytes(b"secret\xff\xfevalue")
    assert tune(gateway, prometheus, "--api-key-file", str(token)) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "it is not UTF-8 text" in err
    assert "secret" not in err
    assert "0xff" not in err


@pytest.mark.parametrize("value", [TWO_LINE_SECRET, "caf\u00e9-secret"])
def test_an_sdk_client_token_that_a_header_cannot_carry_is_refused(
    gateway, prometheus, monkeypatch, value, capsys, caplog
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("SIE_BASE_URL", gateway.url)
    monkeypatch.setenv("SIE_API_KEY", value)
    assert tune(gateway, prometheus) == cli.EXIT_REFUSED
    captured = capsys.readouterr()
    assert "the gateway client token in $SIE_API_KEY must be one run of printable ASCII" in captured.err
    assert_secret_absent(value, captured, caplog)
    assert gateway.requests == []


def test_a_transport_failure_is_reported_without_the_header_it_carried(gateway):
    reader = pools.connect(gateway.url, TWO_LINE_SECRET)
    try:
        with pytest.raises(keda.GatewayError) as raised:
            reader.get_pool("placement-floor-default-l4-default")
    finally:
        reader.close()
    assert str(raised.value) == "the request failed (LocalProtocolError)"
    writer = pools.connect(gateway.url, TWO_LINE_SECRET)
    try:
        with pytest.raises(keda.GatewayError) as raised:
            writer.create_pool(
                "placement-floor-default-l4-default",
                {"l4": 0},
                bundle="default",
                minimum_worker_count=1,
                queue_pool="default",
            )
    finally:
        writer.close()
    assert str(raised.value) == "the request failed (LocalProtocolError)"
    assert gateway.pools == {}


def test_a_token_an_http_header_cannot_encode_is_refused_without_echoing_it(gateway):
    with pytest.raises(ValueError, match=r"^a gateway token holds a character that an HTTP header cannot carry$"):
        pools.connect(gateway.url, "caf\u00e9-secret")


CARRIER = "placement-floor-default-l4-default"
NESTED = b"[" * 100_000


def write_carrier(writer: pools.SdkPools) -> None:
    writer.create_pool(CARRIER, {"l4": 0}, bundle="default", minimum_worker_count=1, queue_pool="default")


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ((200, NESTED), "the gateway returned an unexpected response"),
    ],
)
def test_a_pool_reply_the_applier_cannot_use_fails_the_call(gateway, reply, message):
    gateway.raw_reply = reply
    reader = pools.connect(gateway.url, READ_TOKEN)
    try:
        with pytest.raises(keda.GatewayError) as raised:
            reader.get_pool(CARRIER)
    finally:
        reader.close()
    assert str(raised.value).startswith(message)
    with pytest.raises(keda.GatewayError) as raised:
        write_carrier(pools.connect(gateway.url, ADMIN_TOKEN))
    assert str(raised.value).startswith(message)
    assert [request["method"] for request in gateway.requests] == ["GET", "POST"]


def test_a_deeply_nested_pool_reply_fails_only_the_tick(gateway, prometheus, read_token_file, capsys):
    gateway.raw_reply = (200, NESTED)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file)) == cli.EXIT_FAILED
    errors = [row["error"] for row in rows(capsys.readouterr().out)]
    assert errors == [
        f"reading pool {CARRIER!r} failed: the gateway returned an unexpected response",
        "reading pool 'placement-floor-default-a100-80gb-generation' failed: "
        "the gateway returned an unexpected response",
    ]


def test_a_malformed_pool_reply_fails_only_the_tick(gateway, prometheus, read_token_file, capsys):
    gateway.read_reply = ["not", "a", "pool"]
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file)) == cli.EXIT_FAILED
    errors = [row["error"] for row in rows(capsys.readouterr().out)]
    assert errors == [
        "reading pool 'placement-floor-default-l4-default' failed: the gateway returned an unexpected response",
        "reading pool 'placement-floor-default-a100-80gb-generation' failed: "
        "the gateway returned an unexpected response",
    ]


def test_the_loop_survives_failed_ticks_and_never_renews_a_lease_in_the_background(
    gateway, prometheus, read_token_file, monkeypatch, capsys
):
    monkeypatch.setenv("SIE_ADMIN_TOKEN", ADMIN_TOKEN)
    leases: list[list[str]] = []
    closed: list[pools.SdkPools] = []
    close = pools.SdkPools.close

    def record_close(self: pools.SdkPools) -> None:
        closed.append(self)
        close(self)

    def between_ticks(_seconds: float) -> None:
        leases.append([thread.name for thread in threading.enumerate() if thread.name.startswith("pool-lease-")])
        if len(leases) == 1:
            prometheus.truncate = True
        elif len(leases) == 2:
            prometheus.truncate = False
            gateway.fail_reads = True
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(pools.SdkPools, "close", record_close)
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=time.monotonic, sleep=between_ticks))
    assert run(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply", "--interval-s", "1") == 0
    captured = capsys.readouterr()
    lines = rows(captured.out)
    assert [(row["action"], row["applied"], row["error"] is None) for row in lines] == [
        ("raise_to_floor", True, True),
        ("hold", False, True),
        ("hold", True, True),
        ("hold", False, True),
        (None, False, False),
        (None, False, False),
    ]
    assert "Prometheus series queueDepth: the query failed: IncompleteRead" in captured.err
    assert leases == [[], [], []]
    posts = [request["path"] for request in gateway.requests if request["method"] == "POST"]
    assert posts == ["/v1/pools", "/v1/pools"]
    assert len(closed) == 2
