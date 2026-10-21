"""GPU health probe and recovery from a wedged CUDA context.

A sticky CUDA error (a ``device-side assert``, an illegal memory access, a
launch failure) poisons the process's CUDA context: every later CUDA call
returns the same error, and the CUDA runtime documents only one way back: "To
continue using CUDA, the process must be terminated and relaunched." A
process-alive check never touches CUDA, so without this module ``/healthz``
stays green while every inference fails (issue #1025).

This module:

- probes every device on which the process holds a CUDA context with a tiny
  kernel and a device-to-host copy, on its own stream and thread, so a probe
  neither waits behind inference nor allocates or creates a context of its own;
- latches the wedge the first time a probe fails. ``is_wedged`` then stays true
  and every health surface reports unhealthy without touching CUDA again;
- in a serving process (armed with :func:`exit_on_wedge`), exits with
  :data:`WEDGED_EXIT_CODE` shortly after the latch, so the container runtime
  restarts the worker with a fresh context instead of waiting for a liveness
  probe to give up.

Every probe caller feeds the same latch: ``/livez``, ``/readyz``,
``/ws/status``, the sidecar's IPC ``Ping`` and the IPC server after a batch
fails with an inference error.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final

import torch

logger = logging.getLogger(__name__)

# Re-probing forces a CUDA sync, so cache the result briefly. The /ws/status loop
# calls this every 200ms and the sidecar pings every 2s. A wedged context is
# sticky, so a couple of seconds of staleness is harmless; callers that just saw
# a failure pass ``use_cache=False``.
_PROBE_TTL_SECONDS = 2.0

#: Exit status of a worker that stopped because its CUDA context is wedged
#: (``EX_TEMPFAIL``: a fresh process can succeed).
WEDGED_EXIT_CODE: Final = 75

# Time between latching a wedge and exiting, so responses already being written
# (the failed batch's answer, refusals of later work) reach the sidecar first.
_EXIT_GRACE_SECONDS = 1.0

_state_lock = threading.Lock()
_cached_healthy: bool = True
_cached_at: float = 0.0
_wedged: bool = False
_exit_armed: bool = False
_exit_timer: threading.Timer | None = None

# Every probe runs on this one thread. A probe never queues behind the shared
# default executor, at most one runs at a time, and a hung device blocks only
# this thread (callers then time out, which is what liveness should see).
_probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sie-gpu-probe")
_inflight: Future[bool] | None = None

# Per-device probe stream and scratch tensor, created on first use and only ever
# touched on the probe thread. Reusing the tensor means a probe never allocates,
# so a full GPU cannot make it fail.
_probe_state: dict[str, tuple[torch.cuda.Stream, torch.Tensor]] = {}


def _run_gpu_probe(device: str) -> None:
    """Run one kernel on ``device`` and copy its result to the host.

    The work goes to a dedicated stream, so it does not wait for inference
    kernels queued on other streams. A wedged context raises its sticky error
    here: from the kernel launch, or from the device-to-host copy in ``.item()``
    that waits for it.
    """
    state = _probe_state.get(device)
    if state is None:
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            scratch = torch.zeros(1, device=device)
        state = _probe_state[device] = (stream, scratch)
    stream, scratch = state
    with torch.cuda.stream(stream):
        scratch.fill_(1.0)
        scratch.item()


def _has_cuda_context(index: int) -> bool:
    """Whether this process holds a CUDA context on device ``index``.

    A device without a context has nothing to wedge, and probing it would create
    one: hundreds of MiB of VRAM, and a spurious failure when another process
    (an SGLang engine, for example) already holds the memory. The primary
    context is the one every CUDA library in the process shares.
    """
    has_primary_context = getattr(torch._C, "_cuda_hasPrimaryContext", None)
    if has_primary_context is None:
        return True
    try:
        return bool(has_primary_context(index))
    except Exception:  # noqa: BLE001 - an unknown answer is probed, not skipped
        return True


def _probe_all_devices() -> bool:
    """Probe every visible device that has a context; return whether all ran.

    A worker process can be assigned more than one GPU (``gpu.count > 1``), so a
    wedge on any device must surface. Kubernetes scopes ``device_count()`` to the
    pod's allocated GPUs.

    Returns:
        True if every probed device runs the kernel; False if any device's
        context is wedged. An ``OutOfMemoryError`` (possible only before the
        scratch tensor exists) is treated as healthy: a full device is not a
        wedged one.
    """
    for index in range(torch.cuda.device_count()):
        if not _has_cuda_context(index):
            continue
        try:
            _run_gpu_probe(f"cuda:{index}")
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "GPU health probe hit OutOfMemoryError on cuda:%d; treating as "
                "healthy (memory pressure, not a wedged context).",
                index,
            )
        except Exception:
            # Catch broadly: a sticky device-side assert, a driver/context fault,
            # or any non-RuntimeError CUDA error all mean this device cannot run
            # a trivial kernel. Catching everything also keeps the probe from
            # escaping into health endpoints and the /ws/status loop.
            logger.exception(
                "GPU health probe failed on cuda:%d: the CUDA context is wedged (sticky device error).",
                index,
            )
            return False
    return True


def _exit_process() -> None:
    """Exit at once. The CUDA context is unusable, so no cleanup may touch it."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(WEDGED_EXIT_CODE)


def _start_exit_timer_locked() -> None:
    global _exit_timer
    if _exit_timer is None:
        _exit_timer = threading.Timer(_EXIT_GRACE_SECONDS, _exit_process)
        _exit_timer.daemon = True
        _exit_timer.start()


def _latch_wedged() -> None:
    global _wedged
    with _state_lock:
        if _wedged:
            return
        _wedged = True
        armed = _exit_armed
        if armed:
            _start_exit_timer_locked()
    if armed:
        logger.critical(
            "CUDA context is wedged; refusing new work and exiting with status %d in %.1fs so the worker "
            "restarts with a fresh context.",
            WEDGED_EXIT_CODE,
            _EXIT_GRACE_SECONDS,
        )
    else:
        logger.critical("CUDA context is wedged; refusing new work. Restart this process to recover.")


def _probe_and_record() -> bool:
    global _cached_healthy, _cached_at
    healthy = _probe_all_devices()
    with _state_lock:
        _cached_healthy = healthy
        _cached_at = time.monotonic()
    if not healthy:
        _latch_wedged()
    return healthy


def _submit_probe(*, join_inflight: bool) -> Future[bool]:
    """Schedule a probe on the probe thread.

    A cached-result caller joins a probe that is already running. A caller that
    needs a probe started after its own observation (``use_cache=False``) gets a
    new one, queued behind any running probe.
    """
    global _inflight
    with _state_lock:
        if join_inflight and _inflight is not None and not _inflight.done():
            return _inflight
        _inflight = _probe_executor.submit(_probe_and_record)
        return _inflight


def _cached_result(use_cache: bool) -> bool | None:
    """Answer without probing when possible: wedged, CPU-only, or a fresh cache."""
    if _wedged:
        return False
    if not torch.cuda.is_available():
        return True
    if use_cache and (time.monotonic() - _cached_at) < _PROBE_TTL_SECONDS:
        return _cached_healthy
    return None


def is_wedged() -> bool:
    """Whether a probe has found this process's CUDA context wedged. Never touches CUDA."""
    return _wedged


def gpu_is_healthy(*, use_cache: bool = True) -> bool:
    """Return whether the GPU can actually run a CUDA kernel right now.

    This is the synchronous entry point; it blocks until the probe finishes.
    Async code should call :func:`gpu_is_healthy_async`.

    On CPU-only workers there is no GPU context to wedge, so this returns True.

    Args:
        use_cache: When True (default) reuse a probe result younger than
            ``_PROBE_TTL_SECONDS``. Pass False to run a probe that starts after
            this call.

    Returns:
        True if CUDA is unavailable or every probed device runs the kernel;
        False if any device's CUDA context is wedged (now or earlier).
    """
    cached = _cached_result(use_cache)
    if cached is not None:
        return cached
    return _submit_probe(join_inflight=use_cache).result()


async def gpu_is_healthy_async(*, use_cache: bool = True) -> bool:
    """Async variant of :func:`gpu_is_healthy`; the probe runs on the probe thread.

    The event loop, which also serves inference handlers and the 200ms
    ``/ws/status`` push, never blocks on the device sync. Arguments and return
    value match :func:`gpu_is_healthy`.
    """
    cached = _cached_result(use_cache)
    if cached is not None:
        return cached
    return await asyncio.wrap_future(_submit_probe(join_inflight=use_cache))


def exit_on_wedge() -> None:
    """Make this process exit once a wedge is latched. Called by the serving entry point.

    Nothing else in the process can bring a wedged context back, so a serving
    worker hands recovery to its supervisor (the kubelet restarts the container
    in seconds) instead of answering health checks with 503 until a liveness
    probe gives up. Library and test use leave this unarmed.
    """
    global _exit_armed
    with _state_lock:
        _exit_armed = True
        if _wedged:
            _start_exit_timer_locked()


def reset_gpu_health_cache() -> None:
    """Clear the cached result, the wedge latch and exit arming. For tests."""
    global _cached_healthy, _cached_at, _wedged, _exit_armed, _exit_timer
    with _state_lock:
        _cached_healthy = True
        _cached_at = 0.0
        _wedged = False
        _exit_armed = False
        if _exit_timer is not None:
            _exit_timer.cancel()
            _exit_timer = None
