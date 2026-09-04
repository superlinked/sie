"""Real CPU server and SDK transport smoke with the checked-in weightless bundle."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sie_sdk import SIEClient

MODEL = "sie-fake"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_api(url: str, process: subprocess.Popen | None = None, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("SIE server exited before becoming ready")
        try:
            with SIEClient(url, timeout_s=2) as client:
                if client.list_models():
                    return
        except Exception:  # noqa: BLE001 — bounded readiness polling
            pass
        time.sleep(1)
    raise RuntimeError(f"SIE API did not become ready at {url}")


def smoke_python(url: str) -> None:
    with SIEClient(url, timeout_s=300) as client:
        assert client.list_models()
        result = client.encode(MODEL, {"id": "cpu-smoke", "text": "Hello world"}, output_types=["dense"])
        assert result["id"] == "cpu-smoke"
        assert result["dense"].shape == (384,)
        assert np.isfinite(result["dense"]).all()
        batch = client.encode(
            MODEL, [{"id": "one", "text": "Hello"}, {"id": "two", "text": "World"}], output_types=["dense"]
        )
        assert [item["id"] for item in batch] == ["one", "two"]
        scored = client.score(MODEL, {"text": "query"}, [{"text": "one"}, {"text": "two"}])
        assert len(scored["scores"]) == 2
        generated = client.generate(MODEL, "a prompt", max_new_tokens=16)
        assert generated["text"]
        assert generated["usage"]["completion_tokens"] == 16
    print("Python SDK CPU encode/batch/score/generate passed.")


def smoke_typescript(url: str) -> None:
    subprocess.run(
        ["mise", "exec", "--", "node", "tools/ci/live_typescript.mjs"],
        env={**os.environ, "SIE_SERVER_URL": url},
        check=True,
        timeout=300,
    )


def main() -> None:
    logs = Path(".cache/ci-logs")
    logs.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with (
        tempfile.TemporaryDirectory(prefix="sie-live-", dir="/tmp") as runtime,
        (logs / "live-sdk.log").open("w") as log,
    ):
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "SIE_IPC_SOCKET_PATH": f"{runtime}/worker.sock",
            "SIE_TELEMETRY_DISABLED": "1",
            "HF_HUB_OFFLINE": "1",
            "SIE_FAKE_MEMORY_BUDGET": "4GiB",
        }
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SIE_API_KEY", "SIE_GATEWAY_URL", "SIE_NATS_URL"):
            env.pop(key, None)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sie_server.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "-p",
                str(port),
                "-d",
                "cpu",
                "-b",
                "fake",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_api(url, process)
            smoke_python(url)
            smoke_typescript(url)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            log.flush()
            print((logs / "live-sdk.log").read_text()[-12000:], file=sys.stderr)


if __name__ == "__main__":
    main()
