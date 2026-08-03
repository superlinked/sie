"""Shared fake-stack fixtures: a real `serve -b fake` server subprocess.

Module-scoped on purpose: each test module gets its OWN cold server, so
first-touch (MODEL_LOADING) semantics stay observable per module.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

SLOW_LOAD_S = 3.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def fake_server() -> Iterator[str]:
    port = _free_port()
    env = dict(os.environ)
    env["SIE_FAKE_MEMORY_BUDGET"] = "4GiB"
    env["SIE_FAKE_FAULTS"] = (
        f'{{"sie-fake:small-a": {{"slow_load_s": {SLOW_LOAD_S}}}, "sie-fake": {{"slow_load_s": {SLOW_LOAD_S}}}}}'
    )
    proc = subprocess.Popen(  # noqa: S603 — our own server binary
        [sys.executable, "-m", "sie_server.cli", "serve", "-b", "fake", "-p", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                if httpx.get(f"{base}/readyz", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                proc.terminate()
                msg = "fake server did not become ready"
                raise RuntimeError(msg)
            time.sleep(0.5)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
