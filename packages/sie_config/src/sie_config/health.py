import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Kubernetes liveness probe.

    Returns 200 if the config service process is running.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    """Kubernetes readiness probe.

    Returns 200 if the model registry is loaded.
    """
    if getattr(request.app.state, "model_registry", None) is None:
        raise HTTPException(status_code=503, detail="Model registry not loaded")
    return {"status": "ready"}
