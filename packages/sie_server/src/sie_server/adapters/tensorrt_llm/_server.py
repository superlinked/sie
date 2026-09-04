"""Lifecycle helpers for one local TensorRT-LLM serve subprocess."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_STARTUP_TIMEOUT_S = 900.0
HEALTH_CHECK_INTERVAL_S = 2.0
PROCESS_GROUP_POLL_INTERVAL_S = 0.05
PROCESS_GROUP_KILL_TIMEOUT_S = 5.0
BASE_PORT = 30200
_RESERVED_PORTS: set[int] = set()
_RESERVED_PORTS_LOCK = threading.Lock()


def parse_cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            index = int(device.partition(":")[2])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device {device!r}") from exc
        if index < 0:
            raise ValueError(f"invalid CUDA device {device!r}")
        return index
    raise ValueError("TensorRT-LLM generation requires a CUDA device")


def reserve_port(start_port: int = BASE_PORT) -> int:
    span = 100
    offset = random.randrange(span)  # noqa: S311 - availability spreading, not security
    with _RESERVED_PORTS_LOCK:
        for index in range(span):
            port = start_port + ((offset + index) % span)
            if port in _RESERVED_PORTS:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _RESERVED_PORTS.add(port)
            return port
    raise RuntimeError(f"no free TensorRT-LLM port in {start_port}-{start_port + span - 1}")


def release_port(port: int | None) -> None:
    if port is None:
        return
    with _RESERVED_PORTS_LOCK:
        _RESERVED_PORTS.discard(port)


def open_output_log() -> tempfile._TemporaryFileWrapper:
    return tempfile.NamedTemporaryFile(prefix="trtllm_", suffix=".log", delete=False)


def log_tail(output_file: tempfile._TemporaryFileWrapper | None, *, chars: int = 5000) -> str:
    if output_file is None or chars <= 0:
        return ""
    with contextlib.suppress(OSError):
        output_file.flush()
    try:
        with Path(output_file.name).open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - (chars * 4)))
            return log.read(chars * 4).decode("utf-8", errors="replace")[-chars:]
    except OSError:
        return ""


def launch(
    command: list[str],
    *,
    device_index: int,
    output_file: tempfile._TemporaryFileWrapper,
    environment: dict[str, str],
) -> subprocess.Popen[bytes]:
    child_env = os.environ.copy()
    child_env.update(environment)
    child_env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    return subprocess.Popen(  # noqa: S603 - command is entirely code/profile owned
        command,
        stdout=output_file,
        stderr=subprocess.STDOUT,
        env=child_env,
        start_new_session=True,
    )


def wait_until_ready(
    process: subprocess.Popen[bytes],
    server_url: str,
    *,
    served_model_name: str,
    output_file: tempfile._TemporaryFileWrapper | None,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = f"{server_url}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"trtllm-serve exited during startup: {log_tail(output_file)}")
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                break
        except requests.RequestException:
            # Connection failures are expected while the child is starting.
            pass
        time.sleep(HEALTH_CHECK_INTERVAL_S)
    else:
        raise RuntimeError(f"trtllm-serve did not become healthy: {log_tail(output_file)}")

    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        raise RuntimeError(f"trtllm-serve exhausted its startup budget before warmup: {log_tail(output_file)}")
    warmup = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": served_model_name,
            "prompt": "warmup",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=remaining_s,
    )
    warmup.raise_for_status()
    payload: Any = warmup.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise RuntimeError(f"invalid trtllm-serve warmup response: {json.dumps(payload)[:500]}")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _reap_owned_process_group_children(process_group_id: int) -> None:
    while True:
        try:
            reaped_pid, _status = os.waitpid(-process_group_id, os.WNOHANG)
        except ChildProcessError:
            return
        if reaped_pid == 0:
            return


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes],
    process_group_id: int,
    *,
    deadline: float,
) -> bool:
    while True:
        launcher_exited = process.poll() is not None
        if launcher_exited:
            # The server container runs Python as PID 1, so orphaned
            # TensorRT/MPI descendants are adopted here. Reap killed children
            # in this adapter-owned session before treating zombie-only PGIDs
            # as a failed SIGKILL.
            _reap_owned_process_group_children(process_group_id)
        if not _process_group_exists(process_group_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        interval = min(PROCESS_GROUP_POLL_INTERVAL_S, remaining)
        if not launcher_exited:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=interval)
        else:
            time.sleep(interval)


def terminate(process: subprocess.Popen[bytes] | None, *, timeout_s: float = 10.0) -> None:
    if process is None:
        return
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0)
        return

    term_deadline = time.monotonic() + max(float(timeout_s), 0.0)
    if _wait_for_process_group_exit(process, process_group_id, deadline=term_deadline):
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(term_deadline - time.monotonic(), 0.0))
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)
    kill_deadline = time.monotonic() + PROCESS_GROUP_KILL_TIMEOUT_S
    exited = _wait_for_process_group_exit(process, process_group_id, deadline=kill_deadline)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(kill_deadline - time.monotonic(), 0.0))
    if not exited:
        raise RuntimeError(f"TensorRT-LLM process group {process_group_id} survived SIGKILL")
