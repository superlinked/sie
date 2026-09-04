"""Build and exercise the public CPU containers without registry credentials."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from tools.ci.live_sdk import smoke_python, wait_for_api

SERVICES = ("sie-config", "sie-gateway", "sie-server-sidecar", "sie-mcp", "sie-server-rust-cpu")


def docker(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["docker", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def require_local_docker() -> None:
    context = json.loads(docker("context", "inspect"))[0]
    endpoint = context["Endpoints"]["docker"]["Host"]
    if not os.environ.get("DOCKER_CONTEXT"):
        endpoint = os.environ.get("DOCKER_HOST") or endpoint
    if not endpoint.startswith("unix://"):
        raise RuntimeError(f"CPU smoke requires a local Unix Docker endpoint, got {endpoint}")
    system = json.loads(docker("info", "--format", "{{json .}}"))
    if system["OSType"] != "linux":
        raise RuntimeError("CPU smoke requires a Linux Docker daemon")
    print(f"Local Docker: {endpoint}; Linux {system['Architecture']}")


def wait_health(url: str, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Container health did not become ready: {url}")


def build_images(registry: str, revision: str) -> None:
    common = ["--registry", registry, "--version", "0.0.0", "--source-revision", revision]
    commands = [["build-server", "--platform", "cpu", "--bundle", "default", *common]]
    commands.extend(["build-service", "--service", service, *common] for service in SERVICES)
    for args in commands:
        subprocess.run(["mise", "run", "docker", "--", *args], check=True, timeout=3600)


def main() -> None:
    require_local_docker()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    registry = f"local/sie-ci-{uuid.uuid4().hex[:8]}"
    build_images(registry, revision)
    network = f"sie-ci-{uuid.uuid4().hex[:8]}"
    containers: list[str] = []
    logs = Path(".cache/ci-logs")
    logs.mkdir(parents=True, exist_ok=True)
    docker("network", "create", network)
    with tempfile.TemporaryDirectory(prefix="sie-cpu-ipc-", dir="/tmp") as ipc:

        def start(
            name: str,
            image: str,
            *,
            env: dict[str, str] | None = None,
            command: tuple[str, ...] = (),
            port: int | None = None,
            shared_ipc: bool = False,
        ) -> str:
            container = f"{network}-{name}"
            args = ["run", "--detach", "--name", container, "--network", network, "--network-alias", name]
            if port is not None:
                args += ["-p", f"127.0.0.1::{port}"]
            if shared_ipc:
                args += ["--user", "0:0", "-v", f"{ipc}:/var/run/sie"]
            for key, value in (env or {}).items():
                args += ["-e", f"{key}={value}"]
            containers.append(container)
            docker(*args, image, *command)
            if port is None:
                return container
            host_port = docker("port", container, f"{port}/tcp").rsplit(":", 1)[1]
            return f"http://127.0.0.1:{host_port}"

        def image(service: str) -> str:
            return f"{registry}/{service}:v0.0.0"

        try:
            start("nats", "nats:2.11.8-alpine", command=("-js",))
            config_url = start("config", image("sie-config"), env={"SIE_NATS_URL": "nats://nats:4222"}, port=8080)
            wait_health(f"{config_url}/healthz")
            worker_env = {
                "SIE_POOL": "default",
                "SIE_BUNDLE": "fake",
                "SIE_MACHINE_PROFILE": "cpu",
                "SIE_IPC_SOCKET_PATH": "/var/run/sie/ipc.sock",
                "SIE_TELEMETRY_DISABLED": "1",
                "HF_HUB_OFFLINE": "1",
                "SIE_FAKE_MEMORY_BUDGET": "4GiB",
            }
            worker_url = start(
                "worker",
                f"{registry}/sie-server:v0.0.0-cpu-default",
                env=worker_env,
                command=("serve", "--host", "0.0.0.0", "--port", "8080", "--device", "cpu", "-b", "fake"),
                port=8080,
                shared_ipc=True,
            )
            wait_for_api(worker_url)
            gateway_env = {
                "SIE_NATS_URL": "nats://nats:4222",
                "SIE_CONFIG_SERVICE_URL": "http://config:8080",
                "SIE_GATEWAY_HEALTH_MODE": "nats",
                "SIE_GATEWAY_ENABLE_POOLS": "1",
                "SIE_GATEWAY_REQUEST_TIMEOUT": "60",
                "SIE_GATEWAY_CONFIGURED_GPUS": "cpu",
                "SIE_GATEWAY_CONFIGURED_PHYSICAL_LANES": '[{"pool":"default","machineProfile":"cpu","bundle":"fake"}]',
            }
            gateway_url = start(
                "gateway",
                image("sie-gateway"),
                env=gateway_env,
                command=("--port", "8080", "--host", "0.0.0.0"),
                port=8080,
            )
            sidecar_url = start(
                "sidecar",
                image("sie-server-sidecar"),
                shared_ipc=True,
                port=9095,
                env={
                    **worker_env,
                    "SIE_NATS_URL": "nats://nats:4222",
                    "SIE_WORKER_ID": "cpu-smoke",
                    "SIE_GATEWAY_URL": "http://gateway:8080",
                },
            )
            wait_health(f"{sidecar_url}/readyz")
            wait_for_api(gateway_url)
            smoke_python(gateway_url)
            mcp_url = start(
                "mcp",
                image("sie-mcp"),
                env={
                    "SIE_BASE_URL": "http://gateway:8080",
                    "SIE_MCP_ALLOW_ANONYMOUS": "true",
                    "SIE_MCP_OAUTH_ENABLED": "false",
                },
                port=8088,
            )
            wait_health(f"{mcp_url}/healthz")
            rust_url = start(
                "rust",
                f"{registry}/sie-server-rust:v0.0.0-cpu",
                shared_ipc=True,
                port=8080,
                env={"SIE_IPC_SOCKET_PATH": "/var/run/sie/rust.sock", "SIE_DEVICE": "cpu"},
            )
            wait_health(f"{rust_url}/healthz")
            docker("exec", f"{network}-worker", "python", "-c", IPC_SMOKE)
            print("CPU gateway/config/worker/sidecar queue requests, MCP health and Rust worker IPC passed.")
        finally:
            for container in reversed(containers):
                (logs / f"{container}.log").write_text(docker("logs", container, check=False))
                docker("rm", "--force", container, check=False)
            docker("network", "rm", network, check=False)


IPC_SMOKE = """
import socket, struct, msgpack
def read_exact(stream, size):
    data = b''
    while len(data) < size:
        part = stream.recv(size - len(data))
        assert part, 'IPC closed before the response completed'
        data += part
    return data
with socket.socket(socket.AF_UNIX) as stream:
    stream.settimeout(15)
    stream.connect('/var/run/sie/rust.sock')
    for method in ('Ping', 'WorkerCapabilities'):
        request = msgpack.packb({'version': 1, 'method': method, 'request_id': method,
                                'body': {'timestamp_ms': 1.0} if method == 'Ping' else {}}, use_bin_type=True)
        stream.sendall(struct.pack('>I', len(request)) + request)
        length = struct.unpack('>I', read_exact(stream, 4))[0]
        assert 0 < length < 1048576
        result = msgpack.unpackb(read_exact(stream, length), raw=False)
        assert result['version'] == 1 and result['request_id'] == method and result['ok'], result
        if method == 'Ping':
            assert isinstance(result['body']['ready'], bool), result
            assert result['body']['worker_id'] == 'sie-server-rust', result
        else:
            assert isinstance(result['body']['supported_models'], list), result
print('CPU Rust worker Ping and WorkerCapabilities IPC passed; no inference model configured')
"""


if __name__ == "__main__":
    main()
