"""Shared MLX server subprocess plumbing.

Used by the MLX generation adapter (``generation.py``). Mirrors the SGLang
``_server.py`` patterns (port allocation, subprocess supervision,
health-polling, termination); it launches an ``mlx_lm.server`` child using the
**parent interpreter's own environment**.

No isolated env is needed: generation is served by the device-agnostic
``sglang`` bundle, which installs ``mlx-lm`` (and the ``transformers>=5`` it
requires) into the server's environment on Apple Silicon
(``mlx-lm ; sys_platform == 'darwin'``) — and that bundle carries no
``transformers<5`` embed/rerank stack to conflict with. So the child runs with
``sys.executable``'s ``mlx_lm.server`` console script directly. (On CUDA the
same bundle installs SGLang instead and this adapter is never selected.)

This module deliberately contains no model-specific logic; it just owns the
lifecycle of a single ``mlx_lm.server`` HTTP child process.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

from sie_server.core.load_errors import EngineStartupError

logger = logging.getLogger(__name__)

# In-process record of ports already handed out by ``find_free_port`` but not
# yet bound by their MLX child — mirrors the SGLang TOCTOU mitigation. Uses a
# port range distinct from SGLang's (30000-30099) so the two never collide
# even if both somehow ran in one process.
_RESERVED_PORTS: set[int] = set()
# Loads run on the registry's load-executor thread pool, so two concurrent MLX
# loads could otherwise probe-and-hand-out the same port (TOCTOU). Guard the
# reserved-set scan with a lock, mirroring sglang/_server.py.
_RESERVED_PORTS_LOCK = threading.Lock()
BASE_PORT = 30200

# Model download (Qwen3.5-4B-4bit is ~2.5 GB) plus the first Metal load can take
# minutes on a cold cache. Generous default; override via the env vars below.
DEFAULT_STARTUP_TIMEOUT_S = 900.0
STARTUP_TIMEOUT_ENV_VARS = (
    "SIE_MLX_STARTUP_TIMEOUT_S",
    "SIE_MODEL_READY_TIMEOUT_S",
    "SIE_ADAPTER_STARTUP_TIMEOUT_S",
    "SIE_SERVER_STARTUP_TIMEOUT_S",
)
HEALTH_CHECK_INTERVAL_S = 2.0

ERR_SERVER_STARTUP = "MLX server failed to start within timeout"
ERR_SERVER_CRASH = "MLX server process exited during startup"
ERR_WARMUP_FAILED = "MLX server started but the model failed to warm up (could not serve a test completion)"


def mlx_lm_available() -> bool:
    """True if ``mlx-lm`` is importable in the current (parent) environment.

    The ``sglang`` generation bundle installs it on Apple Silicon; this lets
    :meth:`MLXGenerationAdapter.load` fail with a clear, actionable message
    instead of an opaque "command not found" from the child launch.
    """
    return importlib.util.find_spec("mlx_lm") is not None


def resolve_startup_timeout(timeout_s: float | None = None) -> float:
    """Resolve the MLX startup-health timeout.

    Precedence: explicit value -> env vars in ``STARTUP_TIMEOUT_ENV_VARS`` order
    -> ``DEFAULT_STARTUP_TIMEOUT_S``.
    """
    if timeout_s is not None:
        try:
            value = float(timeout_s)
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value) and value > 0:
            return value
        logger.warning("Ignoring invalid MLX startup timeout override: %r (must be finite > 0)", timeout_s)

    for name in STARTUP_TIMEOUT_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; expected seconds", name, raw)
            continue
        if math.isfinite(value) and value > 0:
            return value
        logger.warning("Ignoring invalid %s=%r; expected finite seconds > 0", name, raw)

    return DEFAULT_STARTUP_TIMEOUT_S


def find_free_port(start_port: int = BASE_PORT) -> int:
    """Find a free port in ``[start_port, start_port + 100)`` (TOCTOU-mitigated)."""
    span = 100
    offset = random.randrange(span)  # noqa: S311 — port selection, not crypto
    with _RESERVED_PORTS_LOCK:
        for i in range(span):
            port = start_port + ((offset + i) % span)
            if port in _RESERVED_PORTS:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _RESERVED_PORTS.add(port)
            return port
    msg = f"Could not find free port in range {start_port}-{start_port + span - 1}"
    raise EngineStartupError(msg)


def release_port(port: int | None) -> None:
    """Return a port handed out by :func:`find_free_port` to the pool.

    Adapters call this from every teardown seam — ``unload()`` and the
    failed-load abort paths — once the MLX child no longer owns the port.
    Idempotent and tolerant: releasing ``None`` or a never-reserved port is
    a no-op, so teardown paths can call it unconditionally. Mirrors
    ``sglang/_server.py``.
    """
    if port is None:
        return
    with _RESERVED_PORTS_LOCK:
        _RESERVED_PORTS.discard(port)


def open_output_log(prefix: str = "mlx_") -> tempfile._TemporaryFileWrapper:
    """Open a named temp file for capturing subprocess stdout/stderr."""
    return tempfile.NamedTemporaryFile(mode="w", prefix=prefix, suffix=".log", delete=False)


def build_launch_command(*, mlx_repo: str, port: int, host: str = "127.0.0.1") -> list[str]:
    """Build the argv that serves ``mlx_repo`` via ``mlx_lm.server`` in the parent env.

    The ``sglang`` (generation) bundle installs ``mlx-lm`` into the server's own
    environment on Apple Silicon, so the child launches with this interpreter's
    ``mlx_lm.server`` console script — no isolated/uv env. Falls back to
    ``python -m mlx_lm server`` if the console script is not on the interpreter's
    ``bin`` path (e.g. an unusual install layout).
    """
    server_bin = Path(sys.executable).parent / "mlx_lm.server"
    launcher = [str(server_bin)] if server_bin.exists() else [sys.executable, "-m", "mlx_lm", "server"]
    return [
        *launcher,
        "--model",
        mlx_repo,
        "--port",
        str(port),
        "--host",
        host,
        "--log-level",
        "INFO",
    ]


def launch_mlx_server(
    cmd: list[str],
    *,
    output_file: tempfile._TemporaryFileWrapper,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch an ``mlx_lm.server`` subprocess in a new process group.

    Args:
        cmd: Full argv (from :func:`build_launch_command`).
        output_file: Temp file open for write — child stdout/stderr is redirected
            here for diagnostics.
        extra_env: Additional environment variables for the subprocess.

    Returns:
        The ``Popen`` handle. ``start_new_session=True`` puts the child in its
        own process group so the whole group can be signalled on shutdown.
    """
    env = os.environ.copy()
    # hf-xet hangs on Apple Silicon (ignores HF download timeouts); force reliable
    # HTTPS for the multi-GB MLX weight download. See cli.py for the parent-process default.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    if extra_env:
        env.update(extra_env)
    logger.info("MLX subprocess output will be logged to: %s", output_file.name)
    return subprocess.Popen(  # noqa: S603 — intentional subprocess call
        cmd,
        env=env,
        stdout=output_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_server(
    server_url: str,
    process: subprocess.Popen[bytes],
    *,
    output_file: tempfile._TemporaryFileWrapper | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Poll the MLX ``/health`` endpoint until the server is ready.

    Returns True if the server reports healthy before the timeout; False if the
    timeout elapses or the subprocess dies. Note: ``mlx_lm.server`` binds the
    HTTP port and answers ``/health`` as soon as httpd is up — the model may
    still be loading. Callers warm the model up with a real request after this
    returns (see :meth:`MLXGenerationAdapter.load`).
    """
    timeout_s = resolve_startup_timeout(timeout_s)
    health_url = f"{server_url}/health"
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout_s:
        if process.poll() is not None:
            logger.error("MLX server exited prematurely with code %s", process.returncode)
            _log_subprocess_output(output_file)
            return False
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass  # Not up yet — keep polling.
        time.sleep(HEALTH_CHECK_INTERVAL_S)

    logger.error("MLX server startup timeout after %gs", timeout_s)
    _log_subprocess_output(output_file)
    return False


def warmup_model(server_url: str, model: str, *, timeout_s: float) -> bool:
    """Force the model load by issuing one tiny non-streaming completion.

    ``mlx_lm.server`` answers ``/health`` as soon as the HTTP server binds — the
    model may still be downloading/loading. A 1-token request blocks until the
    model is actually ready, so :meth:`MLXGenerationAdapter.load` returns only
    once generation works (deterministic readiness; the user's first real
    request is then fast).

    Returns True if the test completion succeeded (the model genuinely serves),
    False otherwise. The caller treats False as a load failure so a child that
    bound the port but cannot serve the model is not reported as "ready".
    """
    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            json={"model": model, "prompt": "ok", "max_tokens": 1, "temperature": 0.0, "stream": False},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        logger.warning("MLX warmup request failed: %s", exc)
        return False
    if response.status_code != 200:
        logger.warning("MLX warmup returned HTTP %d: %s", response.status_code, response.text[:300])
        return False
    logger.info("MLX model %s warmed up and ready", model)
    return True


def _log_subprocess_output(output_file: tempfile._TemporaryFileWrapper | None) -> None:
    if output_file is None:
        return
    try:
        output_file.flush()
    except Exception:  # noqa: BLE001
        return
    try:
        with Path(output_file.name).open() as f:
            output = f.read()
        logger.error("MLX subprocess output from %s:\n%s", output_file.name, output[-5000:])
    except OSError as e:
        logger.error("Failed to read MLX log: %s", e)


def terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    """Terminate the subprocess group: SIGTERM, wait, SIGKILL fallback."""
    if process is None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            # Best-effort: the group is already gone (ProcessLookupError) or
            # outlived even SIGKILL within the grace window (TimeoutExpired).
            # Nothing more to do — the OS reaps it. Logged for post-mortem.
            logger.debug("MLX subprocess group (pid %s) not confirmed dead after SIGKILL", process.pid)
