"""Run one worker process per component of a composite bundle.

A composite bundle (:mod:`sie_server.composite_bundles`) shares one accelerator
between several bundle environments. Kubernetes grants a device to one
container, so the processes share the container that holds it. ``sie-server
serve`` in such an image becomes this supervisor:

- it starts ``sie-server serve`` once per component, with that component's
  dependency overlay (``SIE_BUNDLE_LIBS``), its own IPC socket
  (``<socket>.<component>.sock``) and an HTTP listener on a private Unix socket;
- it answers ``/healthz``, ``/livez`` and ``/readyz`` on the container port,
  with 200 only when every component answers 200, so a wedged component fails
  the container's probes;
- it forwards ``SIGTERM``/``SIGINT`` to every component and, when any component
  exits, stops the others and exits so the kubelet restarts the container.

In the image the supervisor is the container's first process, so its own exit
also ends every component process.

The supervisor imports no serving dependencies and loads no model.
"""

from __future__ import annotations

import contextlib
import http.client
import http.server
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from sie_server.composite_bundles import (
    BUNDLE_LIBS_ENV,
    COMPONENT_ENV,
    COMPONENT_LIBS_ROOT,
    component_libs_dir,
    component_socket_path,
)

logger = logging.getLogger(__name__)

PROBE_PATHS = frozenset({"/healthz", "/livez", "/readyz"})
_HTTP_OK = 200
_HTTP_UNAVAILABLE = 503
IPC_SOCKET_ENV = "SIE_IPC_SOCKET_PATH"
TELEMETRY_INSTANCE_ENV = "SIE_TELEMETRY_INSTANCE_ID"
DEFAULT_PROBE_TIMEOUT_S = 10.0
# The chart gives the pod 65 s and spends 10 s in preStop.
DEFAULT_STOP_GRACE_S = 50.0


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a Unix domain socket."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._unix_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._unix_path)
        self.sock = sock


def probe_component(http_socket: str, path: str, timeout_s: float) -> int:
    """Return the HTTP status a component process answers ``path`` with, or 0."""
    connection = _UnixHTTPConnection(http_socket, timeout_s)
    try:
        connection.request("GET", path, headers={"Host": "localhost"})
        response = connection.getresponse()
        response.read()
        return response.status
    except OSError:
        return 0
    finally:
        connection.close()


def component_environment(
    environ: Mapping[str, str],
    *,
    bundle: str,
    component: str,
    libs_root: str = COMPONENT_LIBS_ROOT,
) -> dict[str, str]:
    """Return the environment one component process runs with."""
    env = dict(environ)
    env["SIE_BUNDLE"] = bundle
    env[COMPONENT_ENV] = component
    libs = component_libs_dir(component, libs_root)
    # Images install every component overlay; a development checkout has none
    # and runs every component in its own environment.
    if Path(libs).is_dir():
        env[BUNDLE_LIBS_ENV] = libs
    if socket_path := env.get(IPC_SOCKET_ENV):
        env[IPC_SOCKET_ENV] = component_socket_path(socket_path, component)
    if instance := env.get(TELEMETRY_INSTANCE_ENV):
        env[TELEMETRY_INSTANCE_ENV] = f"{instance.rstrip('/')}/{component}"
    return env


@dataclass
class _Component:
    name: str
    http_socket: str
    process: subprocess.Popen[bytes] | None = None
    exit_code: int | None = field(default=None)


class CompositeSupervisor:
    """Start, probe and stop the component processes of one composite worker."""

    def __init__(
        self,
        *,
        bundle: str,
        components: Sequence[str],
        child_argv: Sequence[str],
        host: str,
        port: int,
        environ: Mapping[str, str] | None = None,
        libs_root: str = COMPONENT_LIBS_ROOT,
        probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        stop_grace_s: float = DEFAULT_STOP_GRACE_S,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._bundle = bundle
        self._child_argv = list(child_argv)
        self._host = host
        self._port = port
        self._environ = dict(os.environ if environ is None else environ)
        self._libs_root = libs_root
        self._probe_timeout_s = probe_timeout_s
        self._stop_grace_s = stop_grace_s
        self._popen = popen
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="sie-components-"))
        self._components = [
            _Component(name=name, http_socket=str(self._runtime_dir / f"{name}.http.sock")) for name in components
        ]
        self._stop = threading.Event()
        self._server: http.server.ThreadingHTTPServer | None = None

    @property
    def components(self) -> list[str]:
        return [component.name for component in self._components]

    @property
    def probe_port(self) -> int:
        """The TCP port the probes listen on (after :meth:`serve_probes`)."""
        assert self._server is not None
        return int(self._server.server_address[1])

    def start(self) -> None:
        """Start every component process."""
        for component in self._components:
            Path(component.http_socket).unlink(missing_ok=True)
            env = component_environment(
                self._environ, bundle=self._bundle, component=component.name, libs_root=self._libs_root
            )
            argv = [*self._child_argv, "--host", "127.0.0.1", "--uds", component.http_socket]
            logger.info(
                "Starting component %s of bundle %s (IPC socket %s)",
                component.name,
                self._bundle,
                env.get(IPC_SOCKET_ENV, "<none>"),
            )
            component.process = self._popen(argv, env=env)

    def probe(self, path: str) -> dict[str, int]:
        """Probe ``path`` on every component concurrently."""
        with ThreadPoolExecutor(max_workers=len(self._components)) as pool:
            statuses = pool.map(
                lambda component: probe_component(component.http_socket, path, self._probe_timeout_s),
                self._components,
            )
            return dict(zip(self.components, statuses, strict=True))

    def serve_probes(self) -> None:
        """Answer the container's probes on ``host:port`` in a background thread."""
        supervisor = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path not in PROBE_PATHS:
                    self.send_error(404)
                    return
                statuses = supervisor.probe(path)
                healthy = all(status == _HTTP_OK for status in statuses.values())
                body = json.dumps({"status": "ok" if healthy else "unavailable", "components": statuses}).encode()
                self.send_response(_HTTP_OK if healthy else _HTTP_UNAVAILABLE)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("probe %s", format % args)

        self._server = http.server.ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="composite-probes", daemon=True).start()

    def request_stop(self, signum: int | None = None, _frame: FrameType | None = None) -> None:
        if signum is not None:
            logger.info("Received signal %s; stopping components of bundle %s", signum, self._bundle)
        self._stop.set()

    def wait(self, poll_interval_s: float = 0.5) -> int:
        """Block until a stop is requested or a component exits; return the exit code."""
        exited: _Component | None = None
        while not self._stop.is_set():
            for component in self._components:
                assert component.process is not None
                code = component.process.poll()
                if code is not None:
                    component.exit_code = code
                    exited = component
                    break
            if exited is not None:
                logger.error(
                    "Component %s of bundle %s exited with %s; stopping the others",
                    exited.name,
                    self._bundle,
                    exited.exit_code,
                )
                break
            self._stop.wait(poll_interval_s)
        self.stop()
        if exited is None:
            return 0
        # A component that leaves on its own is a failure even with status 0:
        # the container must restart to restore the composite.
        return exited.exit_code or 1

    def stop(self) -> None:
        """Stop every component (SIGTERM, then SIGKILL after the grace period)."""
        alive = [c.process for c in self._components if c.process is not None and c.process.poll() is None]
        for process in alive:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + self._stop_grace_s
        for process in alive:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                process.wait()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        for component in self._components:
            Path(component.http_socket).unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            self._runtime_dir.rmdir()


def run_composite_supervisor(
    *,
    bundle: str,
    components: Sequence[str],
    host: str,
    port: int,
    argv: Sequence[str] | None = None,
) -> int:
    """Supervise the component processes of ``bundle`` until stopped."""
    child_argv = [sys.executable, "-m", "sie_server.cli", *(sys.argv[1:] if argv is None else argv)]
    supervisor = CompositeSupervisor(
        bundle=bundle,
        components=components,
        child_argv=child_argv,
        host=host,
        port=port,
    )
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    supervisor.start()
    supervisor.serve_probes()
    logger.info(
        "Composite bundle %s: %d component processes (%s); probes on %s:%d",
        bundle,
        len(components),
        ", ".join(components),
        host,
        port,
    )
    return supervisor.wait()
