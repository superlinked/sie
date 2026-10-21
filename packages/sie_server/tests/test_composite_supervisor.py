"""The composite supervisor: component processes, probe aggregation, lifecycle."""

from __future__ import annotations

import json
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from sie_server.composite_bundles import BUNDLE_LIBS_ENV, COMPONENT_ENV
from sie_server.composite_supervisor import CompositeSupervisor, component_environment

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix domain sockets")

FAKE_COMPONENT = textwrap.dedent(
    """
    import http.server, os, socketserver, sys, threading, time

    uds = sys.argv[sys.argv.index("--uds") + 1]
    component = os.environ["SIE_BUNDLE_COMPONENT"]
    status = int(os.environ.get(f"FAKE_STATUS_{component}", "200"))
    exit_after = os.environ.get(f"FAKE_EXIT_{component}")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def address_string(self):
            return "uds"

        def log_message(self, *args):
            pass

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    server = Server(uds, Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if exit_after:
        time.sleep(float(exit_after))
        sys.exit(3)
    while True:
        time.sleep(0.1)
    """
)


@pytest.fixture
def fake_component(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_component.py"
    script.write_text(FAKE_COMPONENT)
    return [sys.executable, str(script)]


def _supervisor(fake_component: list[str], **environ: str) -> CompositeSupervisor:
    supervisor = CompositeSupervisor(
        bundle="default-sglang",
        components=["default", "sglang"],
        child_argv=fake_component,
        host="127.0.0.1",
        port=0,
        environ={"PATH": "/usr/bin:/bin", **environ},
        probe_timeout_s=2.0,
        stop_grace_s=5.0,
    )
    supervisor.start()
    supervisor.serve_probes()
    return supervisor


def _get(supervisor: CompositeSupervisor, path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{supervisor.probe_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - local test server
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        body = error.read()
        return error.code, json.loads(body) if body.startswith(b"{") else {}


def _wait_for(supervisor: CompositeSupervisor, path: str, expected: int) -> dict:
    deadline = time.monotonic() + 20
    while True:
        status, body = _get(supervisor, path)
        if status == expected and all(code != 0 for code in body.get("components", {}).values()):
            return body
        if time.monotonic() > deadline:
            pytest.fail(f"{path} never answered {expected}: {status} {body}")
        time.sleep(0.1)


def test_component_environment(tmp_path: Path) -> None:
    (tmp_path / "sglang").mkdir()
    environ = {
        "SIE_IPC_SOCKET_PATH": "/var/run/sie/ipc-1.sock",
        "SIE_TELEMETRY_INSTANCE_ID": "pod-uid/worker-1",
        "SIE_BUNDLE": "ignored",
    }

    sglang = component_environment(environ, bundle="default-sglang", component="sglang", libs_root=str(tmp_path))
    default = component_environment(environ, bundle="default-sglang", component="default", libs_root=str(tmp_path))

    assert sglang["SIE_BUNDLE"] == "default-sglang"
    assert sglang[COMPONENT_ENV] == "sglang"
    assert sglang["SIE_IPC_SOCKET_PATH"] == "/var/run/sie/ipc-1.sglang.sock"
    assert sglang["SIE_TELEMETRY_INSTANCE_ID"] == "pod-uid/worker-1/sglang"
    assert sglang[BUNDLE_LIBS_ENV] == str(tmp_path / "sglang")
    # No overlay installed for this component (a development checkout).
    assert BUNDLE_LIBS_ENV not in default
    assert default["SIE_IPC_SOCKET_PATH"] == "/var/run/sie/ipc-1.default.sock"
    assert environ["SIE_IPC_SOCKET_PATH"] == "/var/run/sie/ipc-1.sock"


def test_probes_pass_only_when_every_component_passes(fake_component: list[str]) -> None:
    supervisor = _supervisor(fake_component, FAKE_STATUS_sglang="503")
    try:
        body = _wait_for(supervisor, "/readyz", 503)
        assert body["components"] == {"default": 200, "sglang": 503}
        assert _get(supervisor, "/metrics")[0] == 404
    finally:
        supervisor.request_stop()
        assert supervisor.wait() == 0

    supervisor = _supervisor(fake_component)
    try:
        for path in ("/healthz", "/livez", "/readyz"):
            assert _wait_for(supervisor, path, 200)["status"] == "ok"
    finally:
        supervisor.request_stop()
        assert supervisor.wait() == 0


def test_a_component_exit_stops_the_others_and_fails_the_container(fake_component: list[str]) -> None:
    supervisor = _supervisor(fake_component, FAKE_EXIT_default="0.5")
    started = time.monotonic()

    code = supervisor.wait(poll_interval_s=0.05)

    assert code == 3
    assert time.monotonic() - started < 15
    for component in supervisor._components:
        assert component.process is not None
        assert component.process.poll() is not None
