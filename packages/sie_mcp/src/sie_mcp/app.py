"""ASGI app for the SIE MCP edge: MCP streamable-HTTP transport + auth + health."""

import logging

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from sie_mcp.auth import ConnectorSecretAuthMiddleware
from sie_mcp.config import MCPConfig
from sie_mcp.oauth import build_oauth_routes
from sie_mcp.server import build_server

logger = logging.getLogger(__name__)


async def _healthz(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_app() -> Starlette:
    """Construct the ASGI app (uvicorn factory target)."""
    config = MCPConfig.from_env()
    app = build_server(config).streamable_http_app()
    app.router.routes.append(Route("/healthz", _healthz, methods=["GET"]))
    if config.oauth_enabled:
        if not config.public_base_url:
            logger.warning(
                "SIE_MCP_PUBLIC_URL is unset: OAuth metadata is served only for loopback or SIE_MCP_ALLOWED_HOSTS "
                "Host headers. Pin SIE_MCP_PUBLIC_URL to the public https origin for any exposed deployment."
            )
        # The OAuth bridge lets claude.ai connectors authenticate via the connector
        # secret; the gate below exempts these bootstrap endpoints.
        app.router.routes.extend(build_oauth_routes(config))
    app.add_middleware(ConnectorSecretAuthMiddleware, config=config)
    return app
