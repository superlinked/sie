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


def run_server(host: str, port: int, reload: bool, config: AppStateConfig) -> None:
    config.save_to_env_vars()
    uvicorn.run(
        "sie_server.main:_create_app_from_env",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        loop="uvloop",
        timeout_keep_alive=120,
    )
