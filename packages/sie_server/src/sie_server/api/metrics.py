"""Prometheus metrics endpoint for SIE Server.

Exposes /metrics endpoint in Prometheus text format for scraping.
"""

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

    Returns metrics in Prometheus text exposition format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
