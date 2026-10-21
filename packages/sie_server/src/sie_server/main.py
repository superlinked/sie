# IMPORTANT: install HF timeout defaults before any transitive import of
# ``huggingface_hub``. Inlined (not imported from a sie_server submodule)
# to avoid triggering ``sie_server.core.__init__`` — which pulls in
# adapters — before the env vars are set. Mirrored from ``cli.py`` to
# cover uvicorn ``reload=True`` worker re-imports (cli is not re-executed
# in workers; they invoke this factory directly).
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

import uvicorn
from fastapi import FastAPI

from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig


def _create_app_from_env() -> FastAPI:
    """Standard FastAPI factory entry point invoked by uvicorn with factory=True.

    This function is called by uvicorn whenever factory=True is passed, which `run_server()`
    always does regardless of whether reload=True or reload=False. It deserializes the
    AppStateConfig from environment variables (set by `run_server()` before starting uvicorn)
    and creates the FastAPI app.
    """
    config = AppStateConfig.from_env_vars()
    return AppFactory.create_app(config)


def run_server(
    host: str,
    port: int,
    reload: bool,
    config: AppStateConfig,
    *,
    uvicorn_log_level: str = "info",
    uds: str | None = None,
) -> None:
    config.save_to_env_vars()
    uvicorn.run(
        "sie_server.main:_create_app_from_env",
        host=host,
        port=port,
        uds=uds,
        reload=reload,
        factory=True,
        loop="uvloop",
        timeout_keep_alive=120,
        log_level=uvicorn_log_level,
    )
