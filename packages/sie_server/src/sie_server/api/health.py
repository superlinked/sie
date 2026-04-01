"""Health check endpoints for SIE Server.

Provides Kubernetes-compatible liveness and readiness probes:
- /healthz: Liveness probe - is the process alive?
- /readyz: Readiness probe - is the server ready to accept traffic?

See DESIGN.md Section 3.1 for specification.
"""

from fastapi import APIRouter, Response

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


@router.get("/readyz")
async def readyz() -> Response:
    """Readiness probe.

    Returns 200 if the server is ready to accept traffic.
    Used by Kubernetes to determine if traffic should be routed to this pod.

    The readiness state is managed by the lifespan handler:
    - Ready after startup completes (all initialization done)
    - Not ready during shutdown (draining in-flight requests)

    Returns:
        200 OK with "ok" body if ready.
        503 Service Unavailable if not ready (starting up or shutting down).
    """
    if not is_ready():
        return Response(content="not ready", status_code=503, media_type="text/plain")
    return Response(content="ok", media_type="text/plain")
