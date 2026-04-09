import logging
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request

from sie_router.proxy import _proxy_request

logger = logging.getLogger(__name__)

# Prefix table — maps path prefix to operation name used in _proxy_request
_HOT_PREFIXES = (
    ("/v1/encode/", "encode"),
    ("/v1/score/", "score"),
    ("/v1/extract/", "extract"),
)


class HotPathMiddleware:
    """Intercept hot proxy paths and handle them without FastAPI overhead.

    Bypasses FastAPI's routing, dependency injection, exception middleware,
    and async exit stack for the three hot proxy endpoints (encode, score,
    extract).  All other paths fall through to the wrapped FastAPI app.

    The middleware extracts the same parameters that FastAPI's DI would
    resolve (model path, two optional headers) via direct scope access,
    constructs a Starlette ``Request``, and calls ``_proxy_request``
    — the same function the FastAPI endpoints delegate to.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method", "GET") != "POST":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]

        # Fast prefix check (~0.1μs) instead of Starlette's regex routing (~1μs)
        operation = None
        model = None
        for prefix, op in _HOT_PREFIXES:
            if path.startswith(prefix):
                operation = op
                model = path[len(prefix) :]
                break

        if operation is None or not model:
            await self.app(scope, receive, send)
            return

        # Extract the two optional headers directly from scope.
        # scope["headers"] is a list of (name, value) byte tuples — iterate
        # once to collect both, avoiding dict() construction overhead.
        gpu: str | None = None
        pool: str | None = None
        for name, value in scope["headers"]:
            if name == b"x-sie-machine-profile":
                gpu = value.decode() or None
            elif name == b"x-sie-pool":
                pool = value.decode() or None

        request = Request(scope, receive)

        try:
            response = await _proxy_request(
                request=request,
                model=model,
                path=f"/v1/{operation}/{model}",
                x_machine_profile=gpu,
                x_sie_pool=pool,
            )
            await response(scope, receive, send)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
            error_response = JSONResponse(status_code=exc.status_code, content=detail)
            await error_response(scope, receive, send)
        except Exception:
            logger.exception("Unhandled error in hot path handler for %s %s", scope.get("method"), path)
            error_response = JSONResponse(
                status_code=500,
                content={"error": "internal_error", "message": "Internal server error"},
            )
            await error_response(scope, receive, send)
