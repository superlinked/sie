"""Run functional Rust tests with mandatory local JetStream coverage."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_jetstream(url: str, process: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("NATS exited before JetStream became ready")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if "streams" in json.load(response):
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Mandatory JetStream broker did not become ready")


def main() -> None:
    if shutil.which("nats-server") is None:
        raise RuntimeError("nats-server is required; run mise install")
    if shutil.which("mise") is None:
        raise RuntimeError("mise is required by the sidecar integration harness")
    port, monitor = free_port(), free_port()
    logs = Path(".cache/ci-logs")
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sie-ci-nats-") as storage, (logs / "nats.log").open("w") as log:
        process = subprocess.Popen(
            ["nats-server", "-js", "-a", "127.0.0.1", "-p", str(port), "-m", str(monitor), "-sd", storage],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_jetstream(f"http://127.0.0.1:{monitor}/jsz", process)
            env = {**os.environ, "NATS_URL": f"nats://127.0.0.1:{port}", "SIE_RUN_NATS_PUBLISHER_TEST": "1"}
            for key in tuple(env):
                if "BENCHMARK" in key:
                    env.pop(key)
            for args in (
                ["test", "--locked", "--workspace"],
                ["test", "--locked", "-p", "sie-server-sidecar", "--features", "cloud-storage"],
                ["test", "--locked", "--manifest-path", "packages/sie_server_rust/Cargo.toml"],
            ):
                subprocess.run(["cargo", *args], env=env, check=True, timeout=2700)
                wait_for_jetstream(f"http://127.0.0.1:{monitor}/jsz", process)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
