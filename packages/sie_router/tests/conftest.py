from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Generator

import pytest

logger = logging.getLogger(__name__)


def _find_free_port(start: int = 14222, end: int = 14322) -> int:
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    msg = f"No free port in range {start}-{end}"
    raise RuntimeError(msg)


@pytest.fixture(scope="session")
def nats_server() -> Generator[str]:
    """Start a local NATS server with JetStream for integration tests.

    If ``SIE_NATS_URL`` is already set the fixture assumes an external server
    is running and yields that URL without starting anything.

    Yields the ``nats://host:port`` URL.
    """
    external_url = os.environ.get("SIE_NATS_URL")
    if external_url:
        logger.info("Using external NATS server: %s", external_url)
        yield external_url
        return

    nats_bin = shutil.which("nats-server")
    if nats_bin is None:
        pytest.skip("nats-server not found in PATH — required for integration tests")

    port = _find_free_port()
    cmd = [nats_bin, "-p", str(port), "-js", "-m", "0"]

    logger.info("Starting NATS server: %s", " ".join(cmd))
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    url = f"nats://127.0.0.1:{port}"
    os.environ["SIE_NATS_URL"] = url

    # Wait for NATS to accept connections
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            if proc.poll() is not None:
                out = proc.stdout.read().decode() if proc.stdout else ""
                pytest.fail(f"nats-server exited early (rc={proc.returncode}):\n{out}")
            time.sleep(0.1)
    else:
        proc.terminate()
        pytest.fail("nats-server failed to accept connections within 10s")

    logger.info("NATS server ready at %s", url)

    try:
        yield url
    finally:
        os.environ.pop("SIE_NATS_URL", None)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logger.info("NATS server stopped")
