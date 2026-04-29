# `/metrics` endpoint -- Prometheus text exposition format.
#
# Kept in its own module so the rest of the service doesn't have to
# import FastAPI/prometheus_client just to get at the metric objects
# (those live in `sie_config.metrics`).

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_class=Response,
    responses={
        200: {
            "description": "Prometheus metrics in text format",
            "content": {"text/plain": {}},
        },
    },
)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping.

    Uses the default `prometheus_client` registry, which is what all
    metric objects in `sie_config.metrics` register into.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
