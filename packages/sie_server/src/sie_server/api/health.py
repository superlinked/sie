"""Health check endpoints for SIE Server.

Provides Kubernetes-compatible probes:
- /healthz: Startup probe - is the process alive? (CPU only, cheap)
- /livez:   GPU-aware liveness probe - can the device still run kernels?
- /readyz:  Readiness probe - is the server ready to accept traffic right now?

A wedged CUDA context does not wait for a probe to give up: the first probe that
finds it (from any of these endpoints, ``/ws/status`` or the sidecar's IPC
``Ping``) makes a serving worker exit so its container restarts
(``sie_server.core.gpu_health``).
"""

from fastapi import APIRouter, Response

from sie_server.core.gpu_health import gpu_is_healthy_async
from sie_server.core.readiness import is_ready

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> Response:
    """Liveness probe.

    Returns 200 if the server process is alive and responding.
    Used by Kubernetes to detect if the container needs to be restarted.

    Returns:
        200 OK with "ok" body.
    """
    return Response(content="ok", media_type="text/plain")


@router.get("/livez")
async def livez() -> Response:
    """GPU-aware liveness probe.

    Unlike /healthz (process-alive only), this runs a tiny kernel on every
    device that holds a CUDA context. The probe has its own stream and thread,
    so neither a long batch nor a model load delays it. A wedged context
    (``device-side assert``) answers 503 and makes a serving worker exit; a hung
    device or event loop never answers, which the kubelet's ``timeoutSeconds``
    turns into a failure.

    It deliberately does NOT consult the lifecycle ready flag, so graceful
    shutdown draining (when the worker marks itself not-ready) never trips a
    restart.

    Returns:
        200 OK with "ok" body if the GPU can run a kernel (or this is a CPU
        worker). 503 Service Unavailable with "gpu unhealthy" if wedged.
    """
    if not await gpu_is_healthy_async():
        return Response(content="gpu unhealthy", status_code=503, media_type="text/plain")
    return Response(content="ok", media_type="text/plain")


@router.get("/readyz")
async def readyz() -> Response:
    """Readiness probe.

    Returns 200 if the server is ready to accept traffic.
    Used by Kubernetes to determine if traffic should be routed to this pod.

    Readiness has two conditions:
    - Lifecycle state, managed by the lifespan handler: ready after startup
      completes, not ready during shutdown (draining in-flight requests).
    - GPU health: a tiny CUDA sync confirms the device can still run kernels. A
      wedged CUDA context (``device-side assert``) keeps returning sticky errors
      from every inference while a process-alive check stays green, so without
      this the gateway would keep routing to a dead worker.

    Returns:
        200 OK with "ok" body if ready.
        503 Service Unavailable if starting up / shutting down ("not ready") or
        the GPU context is wedged ("gpu unhealthy").
    """
    if not is_ready():
        return Response(content="not ready", status_code=503, media_type="text/plain")
    if not await gpu_is_healthy_async():
        return Response(content="gpu unhealthy", status_code=503, media_type="text/plain")
    return Response(content="ok", media_type="text/plain")
