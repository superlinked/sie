import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import api_router

logger = logging.getLogger(__name__)

# Origins allowed for CORS (frontend dev server)
CORS_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}

app = FastAPI(title="SIE LLM Search Backend")


def _add_cors_headers(response: Response, origin: str | None) -> None:
    if origin and origin in CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"


class EnsureCORSHeadersMiddleware(BaseHTTPMiddleware):
    """Add CORS headers to every response so errors (e.g. 500) never bypass CORS."""

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "OPTIONS":
            resp = Response(status_code=200)
            _add_cors_headers(resp, origin)
            return resp
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled error in middleware chain: %s", exc)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "type": type(exc).__name__},
            )
        _add_cors_headers(response, origin)
        return response


app.add_middleware(EnsureCORSHeadersMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return JSON 500 so the real error is logged and client gets a proper body."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def _startup_schema():
    """Create tables (if missing) and add columns introduced after initial schema."""
    from app.config import settings
    from app.db.migrate import ensure_schema

    ensure_schema()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}

