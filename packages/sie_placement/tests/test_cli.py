from __future__ import annotations

import json
import logging
import os
import ssl
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("sie_sdk")

import certifi
import httpx
from sie_placement import cli, keda, pools
from sie_sdk import SIEError

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
        self.raw_replies: dict[str, tuple[int, bytes]] = {}

    def respond(self, handler: _Handler, method: str, body: object) -> None:
        path = urlsplit(handler.path).path
        authorization = handler.headers.get("Authorization")
        self.requests.append({"method": method, "path": path, "authorization": authorization, "body": body})
        raw = self.raw_replies.get(method, self.raw_reply)
        if raw is not None:
            status, payload = raw
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


class FakeProxy(_Server):
    def respond(self, handler: _Handler, method: str, body: object) -> None:
        self.requests.append(
            {"method": method, "target": handler.path, "authorization": handler.headers.get("Authorization")}
        )
        handler._reply(502, {"detail": "this proxy forwards nothing"})


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
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gateway() -> Iterator[FakeGateway]:
    yield from _serve(FakeGateway())


@pytest.fixture
def prometheus() -> Iterator[FakePrometheus]:
    yield from _serve(FakePrometheus())


@pytest.fixture
def proxy() -> Iterator[FakeProxy]:
    yield from _serve(FakeProxy())


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
    assert l4_row["error"] == "writing pool 'placement-floor-default-l4-default' failed: the gateway answered HTTP 403"


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


def test_a_plan_nested_deeper_than_the_parser_allows_is_refused(gateway, prometheus, tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text("[" * 200_000, encoding="utf-8")
    assert tune(gateway, prometheus, plan=plan) == cli.EXIT_REFUSED
    assert "is not JSON: it nests deeper than the parser allows" in capsys.readouterr().err
    assert gateway.requests == []
    assert prometheus.requests == []


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


PASSWORD = "s3cret-password"  # noqa: S105 - a test value


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("sie-gateway:8080", "must be an absolute http or https URL"),
        ("ftp://sie-gateway:8080", "must be an absolute http or https URL"),
        ("http:///v1", "must be an absolute http or https URL"),
        (f"http://user:{PASSWORD}@sie-gateway:8080", "must not embed credentials"),
        (f"http://user:{PASSWORD}@sie-gateway:notaport", "must not embed credentials"),
        (f"http://sie-gateway:8080/?token={PASSWORD}", "must not carry a query or a fragment"),
        (f"http://sie-gateway:8080/#{PASSWORD}", "must not carry a query or a fragment"),
        ("http://sie-gateway:notaport", "has an invalid port"),
        (f"http://user:{PASSWORD}@gw\uff0fx:8080", "is not a valid URL"),
        (f"http://user:{PASSWORD}\uff20gw:8080", "is not a valid URL"),
    ],
)
def test_a_gateway_url_that_is_not_a_plain_base_url_is_refused_without_echoing_it(
    gateway, prometheus, read_token_file, monkeypatch, url, message, capsys, caplog
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("SIE_ADMIN_TOKEN", ADMIN_TOKEN)
    argv = ["tune", "--plan", str(FIXTURES / "tune-plan.json"), "--prometheus-url", prometheus.url]
    argv += ["--gateway-url", url, "--api-key-file", str(read_token_file), "--apply", "--once"]
    assert cli.main(argv) == cli.EXIT_REFUSED
    captured = capsys.readouterr()
    assert f"the gateway URL {message}" in captured.err
    for text in (captured.out, captured.err, caplog.text):
        assert url not in text
        assert PASSWORD not in text
    assert gateway.requests == []
    assert prometheus.requests == []


def test_a_prometheus_url_that_is_not_a_valid_url_is_refused_without_echoing_it(gateway, prometheus, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    url = f"http://user:{PASSWORD}@prometheus\uff0fx:9090"
    argv = ["tune", "--plan", str(FIXTURES / "tune-plan.json"), "--prometheus-url", url]
    assert cli.main([*argv, "--gateway-url", gateway.url, "--once"]) == cli.EXIT_REFUSED
    captured = capsys.readouterr()
    assert "the Prometheus URL is not a valid URL" in captured.err
    for text in (captured.out, captured.err, caplog.text):
        assert url not in text
        assert PASSWORD not in text
    assert gateway.requests == []
    assert prometheus.requests == []


def test_a_gateway_url_httpx_cannot_parse_is_refused_without_echoing_it():
    with pytest.raises(ValueError, match=r"^the gateway URL is not a valid URL$"):
        pools.connect(f"http://user:{PASSWORD}@sie-gateway:notaport", READ_TOKEN)


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
        ((307, json.dumps({"spec": {"name": CARRIER}}).encode()), "the gateway answered HTTP 307"),
        ((302, b"{}"), "the gateway answered HTTP 302"),
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


def test_a_redirected_write_is_not_recorded_as_applied(gateway, prometheus, read_token_file, monkeypatch, capsys):
    monkeypatch.setenv("SIE_ADMIN_TOKEN", ADMIN_TOKEN)
    gateway.auth = False
    original = gateway.respond

    def redirect_writes(handler: _Handler, method: str, body: object) -> None:
        gateway.raw_reply = (307, json.dumps({"spec": body}).encode()) if method == "POST" else None
        original(handler, method, body)

    monkeypatch.setattr(gateway, "respond", redirect_writes)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token_file), "--apply") == cli.EXIT_FAILED
    written = [row for row in rows(capsys.readouterr().out) if row["floor"]]
    assert written
    for row in written:
        assert row["applied"] is False
        assert "the gateway answered HTTP 307" in row["error"]
    assert gateway.pools == {}


BACKSLASH_READ_TOKEN = "read\\token-for-tests"  # noqa: S105 - a test value
BACKSLASH_ADMIN_TOKEN = "admin\\token-for-tests"  # noqa: S105 - a test value
ECHOES = {
    "403-message": lambda token: (
        403,
        json.dumps({"detail": {"code": "FORBIDDEN", "message": f"token not allowed: Bearer {token}"}}).encode(),
    ),
    "403-detail": lambda token: (
        403,
        json.dumps({"detail": {"code": "FORBIDDEN", "seen": f"Bearer {token}"}}).encode(),
    ),
    "502-text": lambda token: (502, f"upstream echoed Authorization: Bearer {token}".encode()),
}


@pytest.mark.parametrize("echo", sorted(ECHOES))
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_a_reply_that_echoes_the_clients_token_is_reported_by_its_status_alone(
    gateway, prometheus, tmp_path, monkeypatch, method, echo, capsys
):
    read_token = tmp_path / "read-token"
    read_token.write_text(BACKSLASH_READ_TOKEN + "\n", encoding="utf-8")
    monkeypatch.setenv("SIE_ADMIN_TOKEN", BACKSLASH_ADMIN_TOKEN)
    gateway.auth = False
    status, body = ECHOES[echo](BACKSLASH_READ_TOKEN if method == "GET" else BACKSLASH_ADMIN_TOKEN)
    gateway.raw_replies[method] = (status, body)
    assert tune(gateway, prometheus, "--api-key-file", str(read_token), "--apply") == cli.EXIT_FAILED
    captured = capsys.readouterr()
    errors = [row["error"] for row in rows(captured.out) if row["error"]]
    assert errors
    assert all(error.endswith(f" failed: the gateway answered HTTP {status}") for error in errors)
    for secret in (BACKSLASH_READ_TOKEN, BACKSLASH_ADMIN_TOKEN):
        for form in (secret, secret.replace("\\", "\\\\")):
            assert form not in captured.out
            assert form not in captured.err


def test_an_error_the_sdk_raises_before_any_reply_is_redacted_and_cut():
    message = f"connection error after sending Bearer {BACKSLASH_ADMIN_TOKEN} " + "x" * 5000
    error = str(pools._gateway_error(SIEError(message), None, BACKSLASH_ADMIN_TOKEN))
    assert BACKSLASH_ADMIN_TOKEN not in error
    assert error.startswith("connection error after sending Bearer [redacted] x")
    assert len(error) == pools.MAX_ERROR_LENGTH


@pytest.mark.parametrize("variable", ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
def test_the_gateway_client_sends_no_token_through_a_proxy_the_environment_names(gateway, proxy, monkeypatch, variable):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, proxy.url)
    reader = pools.connect(gateway.url, READ_TOKEN)
    try:
        assert reader.get_pool(CARRIER) is None
    finally:
        reader.close()
    writer = pools.connect(gateway.url, ADMIN_TOKEN)
    try:
        write_carrier(writer)
    finally:
        writer.close()
    assert proxy.requests == []
    assert [(request["method"], request["authorization"]) for request in gateway.requests] == [
        ("GET", f"Bearer {READ_TOKEN}"),
        ("POST", f"Bearer {ADMIN_TOKEN}"),
    ]


HTTPS_GATEWAY = "https://gateway.example:8443"


def recorded_locations(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def record(self, cafile=None, capath=None, cadata=None):
        calls.append({"cafile": cafile, "capath": capath})

    monkeypatch.setattr(ssl.SSLContext, "load_verify_locations", record)
    return calls


def certificate_directory(path: Path) -> Path:
    path.mkdir()
    (path / "ca.0").write_text("")
    return path


@pytest.mark.parametrize(
    ("variable", "also_set_directory"),
    [("SSL_CERT_FILE", False), ("SSL_CERT_DIR", False), ("SSL_CERT_FILE", True)],
)
def test_an_https_gateway_is_verified_against_the_ca_certificates_the_environment_names(
    monkeypatch, tmp_path, variable, also_set_directory
):
    location = tmp_path / "ca.pem" if variable == "SSL_CERT_FILE" else certificate_directory(tmp_path / "certs")
    monkeypatch.setenv(variable, str(location))
    if also_set_directory:
        monkeypatch.setenv("SSL_CERT_DIR", str(certificate_directory(tmp_path / "other-certs")))
    calls = recorded_locations(monkeypatch)
    context = pools._verification(HTTPS_GATEWAY)
    expected = {"cafile": str(location), "capath": None}
    if variable == "SSL_CERT_DIR":
        expected = {"cafile": None, "capath": str(location)}
    assert calls == [expected]
    assert context.keylog_filename is None


def test_a_ca_directory_of_hash_links_to_certificate_files_is_passed_through(monkeypatch, tmp_path):
    (tmp_path / "ca.pem").write_text("")
    links = tmp_path / "certs"
    links.mkdir()
    (links / "ca.0").symlink_to(tmp_path / "ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", str(links))
    calls = recorded_locations(monkeypatch)
    pools._verification(HTTPS_GATEWAY)
    assert calls == [{"cafile": None, "capath": str(links)}]


def test_a_ca_directory_entry_that_cannot_be_inspected_does_not_hide_a_certificate_file(monkeypatch, tmp_path):
    certs = tmp_path / "certs"
    certs.mkdir()
    (certs / "0000.0").symlink_to(certs / "0000.0")
    (certs / "ffff.0").write_text("")
    scandir = os.scandir

    @contextmanager
    def in_name_order(path):
        with scandir(path) as entries:
            yield iter(sorted(entries, key=lambda entry: entry.name))

    monkeypatch.setattr(os, "scandir", in_name_order)
    monkeypatch.setenv("SSL_CERT_DIR", str(certs))
    calls = recorded_locations(monkeypatch)
    pools._verification(HTTPS_GATEWAY)
    assert calls == [{"cafile": None, "capath": str(certs)}]


def test_a_ca_directory_the_applier_cannot_list_is_left_to_openssl(monkeypatch, tmp_path):
    certs = certificate_directory(tmp_path / "certs")
    certs.chmod(0o311)
    try:
        monkeypatch.setenv("SSL_CERT_DIR", str(certs))
        calls = recorded_locations(monkeypatch)
        pools._verification(HTTPS_GATEWAY)
    finally:
        certs.chmod(0o755)
    assert calls == [{"cafile": None, "capath": str(certs)}]


@pytest.mark.parametrize("location", ["unsearchable", "below-unsearchable-parent"])
def test_a_ca_location_the_applier_may_not_search_is_refused_before_any_call(monkeypatch, tmp_path, location):
    locked = tmp_path / "locked"
    locked.mkdir()
    certs = locked if location == "unsearchable" else locked / "certs"
    locked.chmod(0o000 if location == "unsearchable" else 0o600)
    try:
        monkeypatch.setenv("SSL_CERT_DIR", str(certs))
        with pytest.raises(ValueError, match="the CA certificates that SSL_CERT_DIR names cannot be loaded"):
            pools.connect(HTTPS_GATEWAY, READ_TOKEN)
    finally:
        locked.chmod(0o755)


def test_a_ca_directory_list_with_one_usable_entry_is_passed_through_unchanged(monkeypatch, tmp_path):
    good = certificate_directory(tmp_path / "certs")
    listing = os.pathsep.join([str(tmp_path / "missing"), str(good), ""])
    monkeypatch.setenv("SSL_CERT_DIR", listing)
    calls = recorded_locations(monkeypatch)
    pools._verification(HTTPS_GATEWAY)
    assert calls == [{"cafile": None, "capath": listing}]


@pytest.mark.parametrize("url", [HTTPS_GATEWAY, "http://gateway.example:8080"])
def test_the_gateway_client_uses_the_default_ca_bundle_without_those_variables(monkeypatch, url):
    calls = recorded_locations(monkeypatch)
    pools._verification(url)
    assert calls == [{"cafile": certifi.where(), "capath": None}]


def test_an_http_gateway_reads_no_ca_variables(gateway, monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", os.pathsep.join([str(tmp_path / "missing"), str(tmp_path / "gone")]))
    reader = pools.connect(gateway.url, READ_TOKEN)
    try:
        assert reader.get_pool(CARRIER) is None
    finally:
        reader.close()


@pytest.mark.parametrize("url", [HTTPS_GATEWAY, "http://gateway.example:8080"])
def test_the_gateway_client_verifies_as_httpx_does_and_never_writes_tls_secrets_to_a_key_log(
    monkeypatch, tmp_path, url
):
    key_log = tmp_path / "keys.log"
    monkeypatch.setenv("SSLKEYLOGFILE", str(key_log))
    assert ssl.create_default_context().keylog_filename == str(key_log)
    context = pools._verification(url)
    reference = httpx.create_ssl_context()
    assert context.keylog_filename is None
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True
    settings = ("verify_mode", "check_hostname", "minimum_version", "verify_flags", "options", "security_level")
    assert {name: getattr(context, name) for name in settings} == {name: getattr(reference, name) for name in settings}


def test_the_gateway_client_sets_the_strict_verify_flags_a_newer_default_context_sets(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
    flags = pools._verification(HTTPS_GATEWAY).verify_flags
    assert flags & ssl.VERIFY_X509_STRICT
    assert flags & ssl.VERIFY_X509_PARTIAL_CHAIN


@pytest.mark.parametrize("key_log", ["directory", "missing-parent"])
def test_a_key_log_the_client_cannot_open_neither_stops_it_nor_receives_anything(
    gateway, monkeypatch, tmp_path, key_log
):
    directory = tmp_path / "keys"
    directory.mkdir()
    path = directory if key_log == "directory" else tmp_path / "missing" / "keys.log"
    monkeypatch.setenv("SSLKEYLOGFILE", str(path))
    with pytest.raises(IsADirectoryError if key_log == "directory" else FileNotFoundError):
        ssl.create_default_context()
    assert pools._verification(HTTPS_GATEWAY).keylog_filename is None
    pools.connect(HTTPS_GATEWAY, READ_TOKEN).close()
    reader = pools.connect(gateway.url, READ_TOKEN)
    try:
        assert reader.get_pool(CARRIER) is None
    finally:
        reader.close()
    assert list(directory.iterdir()) == []
    assert not (tmp_path / "missing").exists()


def test_a_ca_bundle_that_cannot_be_loaded_is_refused_before_any_call(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    with pytest.raises(ValueError, match="the CA certificates that SSL_CERT_FILE names cannot be loaded"):
        pools.connect(HTTPS_GATEWAY, READ_TOKEN)


@pytest.mark.parametrize("url", [HTTPS_GATEWAY, "http://gateway.example:8080"])
def test_a_default_ca_bundle_that_cannot_be_loaded_is_refused_by_its_path(monkeypatch, tmp_path, url):
    bundle = tmp_path / "missing.pem"
    monkeypatch.setattr(certifi, "where", lambda: str(bundle))
    with pytest.raises(ValueError, match=r"^the default CA bundle ") as refused:
        pools.connect(url, READ_TOKEN)
    assert str(refused.value) == f"the default CA bundle {bundle} cannot be loaded"


@pytest.mark.parametrize("directory", ["missing", "empty", "file", "list", "subdirectory", "dangling-link"])
def test_a_ca_directory_that_holds_no_certificates_is_refused_before_any_call(monkeypatch, tmp_path, directory):
    location = tmp_path / "certs"
    value = str(location)
    if directory in {"empty", "subdirectory", "dangling-link"}:
        location.mkdir()
    if directory == "subdirectory":
        (location / "nested").mkdir()
    elif directory == "dangling-link":
        (location / "ca.0").symlink_to(tmp_path / "missing.pem")
    elif directory == "file":
        location.write_text("")
    elif directory == "list":
        (tmp_path / "empty").mkdir()
        value = os.pathsep.join([str(location), str(tmp_path / "empty")])
    monkeypatch.setenv("SSL_CERT_DIR", value)
    with pytest.raises(ValueError, match="the CA certificates that SSL_CERT_DIR names cannot be loaded"):
        pools.connect(HTTPS_GATEWAY, READ_TOKEN)


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
