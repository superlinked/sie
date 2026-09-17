"""Shared SGLang server subprocess plumbing.

Used by both the embedding adapter (``embedding.py``) and the generation
adapter (``generation.py``). The two adapters launch the same
``sglang.launch_server`` binary with different flags, but the
port-allocation, subprocess-supervision, health-polling, and termination
patterns are identical — hence this module.

This module deliberately contains no model-specific logic; it just owns the
lifecycle of a single SGLang HTTP server child process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests

from sie_server.config.device_groups import (
    MAX_TENSOR_PARALLEL_SIZE,
    format_device_mask,
    resolve_device_group,
    validate_tensor_parallel_size,
)
from sie_server.core.oom import is_oom_error

__all__ = [
    "MAX_TENSOR_PARALLEL_SIZE",
    "format_device_mask",
    "resolve_device_group",
    "validate_tensor_parallel_size",
]

logger = logging.getLogger(__name__)

# In-process record of ports already handed out by ``find_free_port`` but
# not yet bound by their SGLang child. ``find_free_port`` only confirms a
# port is bindable *now*; the caller then closes the probe socket and the
# child binds it moments later — a TOCTOU window. Two concurrent loads in
# the same worker process could otherwise probe-and-hand-out the same port
# (both probes succeed because neither child has bound yet). Recording the
# handed-out port and excluding it on subsequent calls closes the common
# in-process case. Guarded by ``_RESERVED_PORTS_LOCK`` because loads can
# run from different threads (registry load executor).
_RESERVED_PORTS: set[int] = set()
_RESERVED_PORTS_LOCK = threading.Lock()

# 8B+ models can take 5+ min just to download from HF on a fresh cache,
# plus SGLang itself then loads the model onto the GPU. Override via the
# SGLang-specific env var below, or one of the adapter/server aliases that
# operators commonly try when tuning cold-start budgets.
DEFAULT_STARTUP_TIMEOUT_S = 900.0
STARTUP_TIMEOUT_ENV_VARS = (
    "SIE_SGLANG_STARTUP_TIMEOUT_S",
    "SIE_MODEL_READY_TIMEOUT_S",
    "SIE_ADAPTER_STARTUP_TIMEOUT_S",
    "SIE_SERVER_STARTUP_TIMEOUT_S",
)
LIVENESS_BUDGET_ENV_VAR = "SIE_WORKER_LIVENESS_BUDGET_S"
KERNEL_CACHE_ROOT_ENV_VAR = "SIE_SGLANG_KERNEL_CACHE_ROOT"
STARTUP_TIMEOUT_S = DEFAULT_STARTUP_TIMEOUT_S
HEALTH_CHECK_INTERVAL_S = 2.0
BASE_PORT = 30000  # Starting port for SGLang servers
# Collective rendezvous ports for multi-rank groups. The engine otherwise picks
# a random port, so two groups starting together on one host can pick the same
# one and one of them hangs in rendezvous rather than failing. Kept clear of the
# SGLang HTTP span above and of 30200-30299, which the MLX and TensorRT-LLM
# adapters hand out from reservation sets of their own.
NCCL_BASE_PORT = 30400

ERR_SERVER_STARTUP = "SGLang server failed to start within timeout"
ERR_SERVER_CRASH = "SGLang server process exited during startup"
STARTUP_LOG_TAIL_CHARS = 5000
LOAD_HEADROOM_BYTES = 1024**3

_KERNEL_CACHE_LAYOUT_VERSION = "v1"
_JIT_ABI_PACKAGES = (
    "apache-tvm-ffi",
    "cuda-python",
    "flashinfer-cubin",
    "flashinfer-python",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-nvrtc-cu13",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
    "sgl-deep-gemm",
    "sglang",
    "sglang-kernel",
    "torch",
    "triton",
    "xgrammar",
)
_KERNEL_CACHE_DIRS = {
    "CUDA_CACHE_PATH": "cuda",
    "CUTE_DSL_CACHE_DIR": "cutlass",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "SGLANG_CACHE_DIR": "sglang",
    "SGLANG_DG_CACHE_DIR": "deep-gemm",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "XDG_CACHE_HOME": "xdg",
}


def _installed_jit_abi_key() -> str:
    components = [f"python={sys.version_info.major}.{sys.version_info.minor}"]
    for package in _JIT_ABI_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        components.append(f"{package}={version}")
    return hashlib.sha256("\n".join(components).encode()).hexdigest()[:20]


def _gpu_cache_key(device_indices: Sequence[int]) -> str | None:
    """Return one cache key for the whole device group, or None to disable the cache.

    Every member of a tensor-parallel group must be the same product. A mixed
    group is refused rather than cached under one member's name, because the
    artifacts compiled for it are not valid for the others.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        device_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        group_names = [device_names[index] for index in device_indices]
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "SGLang kernel cache disabled: could not identify %s without CUDA init: %s",
            ", ".join(f"cuda:{index}" for index in device_indices),
            exc,
        )
        return None

    if len(set(group_names)) > 1:
        logger.warning(
            "SGLang kernel cache disabled: device group spans more than one product: %s",
            ", ".join(sorted(set(group_names))),
        )
        return None

    device_digest = hashlib.sha256(group_names[0].encode()).hexdigest()[:12]
    return f"gpu-{device_digest}"


def _kernel_cache_env(env: dict[str, str], *, device_indices: Sequence[int]) -> dict[str, str]:
    """Return missing upstream cache variables for one persistent cache root.

    Cache namespaces include the installed JIT ABI and exact GPU product name. This
    prevents a retained local/PVC cache from serving artifacts compiled by a
    different SGLang/Torch/CUDA closure or GPU class. Explicit upstream cache
    variables always win, and any cache setup failure falls back to SGLang's
    ordinary container-local compilation path.
    """
    raw_root = env.get(KERNEL_CACHE_ROOT_ENV_VAR, "").strip()
    if not raw_root:
        return {}

    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        logger.warning("SGLang kernel cache disabled: %s must be an absolute path", KERNEL_CACHE_ROOT_ENV_VAR)
        return {}

    gpu_key = _gpu_cache_key(device_indices)
    if gpu_key is None:
        return {}

    # The group, not the anchor, names the namespace. Artifacts compiled for a
    # four-rank launch are not interchangeable with single-device artifacts.
    group_key = "device-" + "_".join(str(index) for index in device_indices)
    namespace = root / _KERNEL_CACHE_LAYOUT_VERSION / _installed_jit_abi_key() / gpu_key / group_key
    defaults = {name: str(namespace / suffix) for name, suffix in _KERNEL_CACHE_DIRS.items() if not env.get(name)}
    if not defaults:
        return {}
    try:
        for path in defaults.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=namespace, prefix=".sie-write-probe-"):
            pass
    except OSError as exc:
        logger.warning("SGLang kernel cache disabled: cannot prepare %s: %s", namespace, exc)
        return {}
    logger.info("SGLang kernel cache enabled at %s", namespace)
    return defaults


def _resolve_liveness_budget() -> float | None:
    raw = os.environ.get(LIVENESS_BUDGET_ENV_VAR)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected seconds", LIVENESS_BUDGET_ENV_VAR, raw)
        return None
    if math.isfinite(value) and value > 0:
        return value
    logger.warning("Ignoring invalid %s=%r; expected finite seconds > 0", LIVENESS_BUDGET_ENV_VAR, raw)
    return None


def _validate_liveness_budget(timeout_s: float) -> float:
    budget_s = _resolve_liveness_budget()
    if budget_s is not None and timeout_s >= budget_s:
        msg = (
            f"SGLang startup timeout {timeout_s:g}s must be lower than "
            f"{LIVENESS_BUDGET_ENV_VAR}={budget_s:g}s so kubelet liveness does not kill the worker mid-load"
        )
        raise ValueError(msg)
    return timeout_s


def resolve_startup_timeout(timeout_s: float | None = None) -> float:
    """Resolve the SGLang startup-health timeout.

    Precedence:
    1. Explicit adapter/profile value (``adapter_options.loadtime.startup_timeout_s``).
    2. Environment variables in ``STARTUP_TIMEOUT_ENV_VARS`` order.
    3. ``DEFAULT_STARTUP_TIMEOUT_S``.

    Raises:
        ValueError: If an explicit value is not a finite number greater than
            zero, or the resolved timeout is not below the worker liveness
            budget.
    """
    declared = validate_optional_positive(timeout_s, field="startup_timeout_s")
    if declared is not None:
        return _validate_liveness_budget(declared)

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
            return _validate_liveness_budget(value)
        logger.warning("Ignoring invalid %s=%r; expected finite seconds > 0", name, raw)

    return _validate_liveness_budget(DEFAULT_STARTUP_TIMEOUT_S)


def find_free_port(start_port: int = BASE_PORT) -> int:
    """Find a free port in ``[start_port, start_port + 100)``.

    Mitigates the TOCTOU race between probing a port here and the SGLang
    child binding it later: ports handed out by a previous (not-yet-bound)
    call are excluded via ``_RESERVED_PORTS`` so concurrent in-process
    loads can't both pick the same one. The scan start is also randomized
    within the range so two near-simultaneous calls are unlikely to probe
    the same port in the same order. The race against *external* processes
    (outside this interpreter) remains inherent — there is no way to
    atomically reserve a TCP port without holding it open — but the common
    in-process collision is closed. Callers must return the port via
    :func:`release_port` once the child no longer owns it (unload or a
    failed launch); otherwise the 100-port span exhausts under the
    registry's LRU eviction→reload churn and every subsequent load fails
    until the process restarts.
    """
    span = 100
    offset = random.randrange(span)  # noqa: S311 — port selection, not crypto
    with _RESERVED_PORTS_LOCK:
        for i in range(span):
            port = start_port + ((offset + i) % span)
            if port in _RESERVED_PORTS:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                except OSError:
                    continue
            _RESERVED_PORTS.add(port)
            return port
    msg = f"Could not find free port in range {start_port}-{start_port + span - 1}"
    raise RuntimeError(msg)


def reserve_port(port: int) -> None:
    """Reserve one specific port, as :func:`find_free_port` reserves the one it picks.

    For a port a profile declares. Release it with :func:`release_port`.

    Raises:
        RuntimeError: If another model in this process has it reserved, or
            something already has it bound.
    """
    with _RESERVED_PORTS_LOCK:
        if port in _RESERVED_PORTS:
            msg = f"port {port} is already reserved by another model in this process"
            raise RuntimeError(msg)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
            except OSError as error:
                msg = f"port {port} is already in use: {error}"
                raise RuntimeError(msg) from error
        _RESERVED_PORTS.add(port)


def release_port(port: int | None) -> None:
    """Return a port handed out by :func:`find_free_port` to the pool.

    Adapters call this from every teardown seam — ``unload()`` and the
    failed-load abort paths — once the SGLang child no longer owns the port.
    Idempotent and tolerant: releasing ``None`` or a never-reserved port is
    a no-op, so teardown paths can call it unconditionally.
    """
    if port is None:
        return
    with _RESERVED_PORTS_LOCK:
        _RESERVED_PORTS.discard(port)


def parse_device_index(device: str) -> int:
    """Parse device index from device string (e.g. ``"cuda:0"`` → ``0``)."""
    if device in {"cuda", "cpu"}:
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return 0


def _read_cached_model_config(model_name_or_path: str, revision: str | None) -> dict[str, Any] | None:
    """Return a model's ``config.json`` if it can be read without a download.

    Best effort by design. A model served from a local directory, a cached
    repository or neither are all ordinary, so an unreadable config disables
    the checks that depend on it rather than failing a load that would
    otherwise work.
    """
    local = Path(model_name_or_path)
    candidate: Path | None = None
    if local.is_dir():
        candidate = local / "config.json"
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            # HFValidationError subclasses ValueError, so a malformed repo id
            # is covered without importing the hub's error module here.
            cached = try_to_load_from_cache(model_name_or_path, "config.json", revision=revision)
        except (ImportError, OSError, ValueError):
            return None
        if isinstance(cached, str):
            candidate = Path(cached)
    if candidate is None or not candidate.is_file():
        return None
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_width_against_model_config(
    width: int,
    *,
    model_name_or_path: str,
    revision: str | None = None,
) -> None:
    """Refuse a width the model's own shape cannot be divided by.

    The engine asserts these itself, but only after loading weights, which on a
    multi-accelerator load means several cards are held for minutes before the
    launch fails, and a supervisor that restarts the worker repeats that on every restart.

    Only heads the config actually reports are checked, and an unreadable
    config checks nothing, so this can only turn a later crash into an earlier
    and clearer error, never reject a shape that would have worked.

    Raises:
        ValueError: Naming the attribute, its value and the width.
    """
    if width < 2:
        return
    config = _read_cached_model_config(model_name_or_path, revision)
    if config is None:
        return
    # A multimodal config's top-level head counts can describe the vision
    # tower, which is not the stack the engine shards.
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = text_config

    attention_heads = _positive_int(config.get("num_attention_heads"))
    if attention_heads is not None and attention_heads % width:
        msg = (
            f"tensor_parallel_size={width} does not divide num_attention_heads={attention_heads} "
            f"for {model_name_or_path!r}. The engine shards query heads across ranks and "
            "requires an exact division."
        )
        raise ValueError(msg)

    # Key/value heads are replicated rather than split when a model publishes
    # fewer of them than the width, so the engine requires divisibility in
    # whichever direction applies and a grouped-query shape narrower than the
    # group is legal.
    kv_heads = _positive_int(config.get("num_key_value_heads"))
    if kv_heads is not None and kv_heads % width and width % kv_heads:
        msg = (
            f"tensor_parallel_size={width} is neither a multiple nor a divisor of "
            f"num_key_value_heads={kv_heads} for {model_name_or_path!r}. The engine splits "
            "key/value heads across ranks when there are at least as many as the width and "
            "replicates them otherwise, and neither is possible at this width."
        )
        raise ValueError(msg)


def _positive_int(value: object) -> int | None:
    """A config attribute usable as a head count, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def validate_optional_positive(value: float | None, *, field: str) -> float | None:
    """Return a positive finite float, or None.

    Raises:
        ValueError: If the value is present but not a positive finite number.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field} must be a number, got {value!r}"
        raise ValueError(msg)
    if not math.isfinite(value) or value <= 0:
        msg = f"{field} must be a finite number greater than zero, got {value!r}"
        raise ValueError(msg)
    return float(value)


def validate_optional_port(value: int | None) -> int | None:
    """Return a usable TCP port, or None to let SIE reserve one.

    Raises:
        ValueError: If the value is present but not a port number.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1024 <= value <= 65535:
        msg = f"nccl_port must be an integer between 1024 and 65535, got {value!r}"
        raise ValueError(msg)
    return value


def open_output_log(prefix: str = "sglang_") -> tempfile._TemporaryFileWrapper:
    """Open a named temp file for capturing subprocess stdout/stderr."""
    return tempfile.NamedTemporaryFile(
        mode="w",
        prefix=prefix,
        suffix=".log",
        delete=False,
    )


def launch_sglang_server(
    cmd: list[str],
    *,
    device_indices: Sequence[int],
    output_file: tempfile._TemporaryFileWrapper,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch an SGLang HTTP server subprocess over an ordered device group.

    Args:
        cmd: Full argv (must already include ``python -m sglang.launch_server``
            plus all flags).
        device_indices: Ordered CUDA device indices the child may see. A
            single-element sequence is the ordinary one-GPU case. The order is
            preserved, because rank N binds the Nth entry of the mask.
        output_file: Temp file open for write — subprocess stdout/stderr is
            redirected here for debugging.
        extra_env: Additional environment variables to set on the subprocess.
            Used by callers that need to set sglang-specific env knobs (e.g.
            ``SGLANG_ENABLE_SPEC_V2=1`` for NEXTN-on-hybrid-architecture
            models like Qwen3.5-4B).

    Returns:
        The ``Popen`` handle. Subprocess is started in a new process group
        (``start_new_session=True``) so the entire group can be signalled on
        shutdown without affecting the parent.

    Raises:
        ValueError: If no device is supplied, or a device repeats. A repeated
            index would give two ranks the same card and deadlock the group.
    """
    mask = format_device_mask(device_indices)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # The device mask is written after the profile environment, never before.
    # It is the registry's placement decision, and a profile that also set
    # CUDA_VISIBLE_DEVICES would otherwise silently move or widen the claim
    # the registry is accounting for.
    env["CUDA_VISIBLE_DEVICES"] = mask
    for name, value in _kernel_cache_env(env, device_indices=device_indices).items():
        env.setdefault(name, value)
    logger.info("SGLang subprocess output will be logged to: %s", output_file.name)
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
    """Poll the SGLang ``/health`` endpoint until the server is ready.

    Returns:
        True if the server reports healthy before the timeout; False if the
        timeout elapses or the subprocess dies. Subprocess output (when
        ``output_file`` is provided) is logged on failure for diagnostics.
    """
    timeout_s = resolve_startup_timeout(timeout_s)
    health_url = f"{server_url}/health"
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout_s:
        # Check if process died.
        if process.poll() is not None:
            exit_code = process.returncode
            logger.error("SGLang server exited prematurely with code %s", exit_code)
            _log_subprocess_output(output_file)
            return False

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            # Health endpoint not up yet — keep polling until timeout.
            pass

        time.sleep(HEALTH_CHECK_INTERVAL_S)

    logger.error("SGLang server startup timeout after %ds", timeout_s)
    _log_subprocess_output(output_file)
    return False


def _log_subprocess_output(output_file: tempfile._TemporaryFileWrapper | None) -> None:
    output = read_subprocess_output_tail(output_file)
    if output:
        log_path = getattr(output_file, "name", "<unknown>")
        logger.error("SGLang subprocess output from %s:\n%s", log_path, output)


def read_subprocess_output_tail(output_file: tempfile._TemporaryFileWrapper | None) -> str:
    if output_file is None:
        return ""
    try:
        output_file.flush()
    except Exception:  # noqa: BLE001
        return ""
    try:
        with Path(output_file.name).open(encoding="utf-8", errors="replace") as f:
            output = f.read()
        return output[-STARTUP_LOG_TAIL_CHARS:]
    except OSError as exc:
        return f"<failed to read SGLang log: {exc}>"


def startup_failure_error(
    output_file: tempfile._TemporaryFileWrapper | None,
    crash_exit_code: int | None = None,
) -> RuntimeError:
    """Build the startup-failure error, keeping crash and timeout distinct.

    A child process that died must not be reported as a timeout: the loader
    reclassifies timeout-shaped messages as ModelLoadTimeoutError stamped with
    the elapsed time, so a 16.5s engine crash surfaces as "configured=16s"
    while the real budget was 1800s (run 32945082497) and every triage starts
    from a fictional number. Callers pass the pre-terminate ``poll()`` result
    as ``crash_exit_code``; None means the health poll genuinely timed out.
    """
    prefix = f"{ERR_SERVER_CRASH} (exit code {crash_exit_code})" if crash_exit_code is not None else ERR_SERVER_STARTUP
    output = read_subprocess_output_tail(output_file).strip()
    if output and is_oom_error(RuntimeError(output)):
        return RuntimeError(f"{prefix}: out of memory detected in startup log")
    return RuntimeError(prefix)


def estimate_load_required_memory_bytes(
    *,
    device_type: str,
    device_total_bytes: int,
    mem_fraction_static: float,
) -> int | None:
    """Estimate free memory needed before launching a SGLang server."""
    if device_type != "cuda" or device_total_bytes <= 0:
        return None
    if isinstance(mem_fraction_static, bool) or not isinstance(mem_fraction_static, int | float):
        return None
    fraction = float(mem_fraction_static)
    if not 0.0 < fraction <= 1.0:
        return None
    required = int(device_total_bytes * fraction) + LOAD_HEADROOM_BYTES
    return min(required, device_total_bytes)


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
            pass
