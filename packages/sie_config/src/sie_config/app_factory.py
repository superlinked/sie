import logging
import os
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from sie_config import metrics as sie_metrics
from sie_config.config_api import router as config_router
from sie_config.config_store import ConfigStore
from sie_config.health import router as health_router
from sie_config.metrics_endpoint import router as metrics_router
from sie_config.model_registry import ModelRegistry
from sie_config.nats_publisher import NatsPublisher

logger = logging.getLogger(__name__)

# Default paths for bundle and model configs.
# In Docker, SIE_BUNDLES_DIR and SIE_MODELS_DIR are always set (see Dockerfile).
# In development, fall back to the sibling sie_server package in the source tree.
_DEFAULT_BUNDLES_DIR = Path(__file__).parent.parent.parent.parent / "sie_server" / "bundles"
_DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "sie_server" / "models"


class _PrometheusHTTPMiddleware(BaseHTTPMiddleware):
    # Increment `sie_config_http_requests_total` and observe
    # `sie_config_http_request_duration_seconds` for every request.
    #
    # The `path` label is the FastAPI route template (e.g.
    # `/v1/configs/models/{model_id}`) rather than the raw URL so
    # per-model reads do not explode the label cardinality. We resolve
    # the route after the downstream handler runs -- Starlette only
    # populates `request.scope["route"]` once routing has matched. If
    # no route matched (unknown URL -> 404), we fall back to the literal
    # path, which is acceptable because unmatched paths from a known
    # caller set (gateway + admin tooling) are few.
    #
    # We also skip recording for the `/metrics` endpoint itself so the
    # scrape traffic does not pollute the metric it's scraping.
    #
    # The exception path matters as much as the success one. If
    # `call_next(...)` raises, Starlette converts the exception into a
    # 500 *outside* this middleware, so a naive `response = await
    # call_next(request)` followed by `.labels(...)` would miss exactly
    # the failures operators care about most. We wrap the entire
    # critical section in `try / except / finally` so that:
    #   - a raised exception still bumps `status="500"` in the counter
    #     and observes latency (the timing is meaningful -- a 50 ms
    #     crash is very different from a 30 s one),
    #   - the original exception is re-raised so Starlette's default
    #     500 handler runs and the client sees the same response it
    #     would without this middleware.

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        # `status_label` is seeded at "500" so an exception raised
        # inside `call_next` (before we can read `response.status_code`)
        # still attributes the failure correctly in the `finally` block.
        # FastAPI/Starlette's outer exception handler turns the raised
        # exception into a 500 for the client, so counting it as a 500
        # matches what the caller actually sees.
        start = time.monotonic()
        status_label = "500"
        try:
            response = await call_next(request)
            status_label = str(response.status_code)
            return response
        finally:
            elapsed = time.monotonic() - start
            route = request.scope.get("route")
            path_label = getattr(route, "path", None) or request.url.path

            sie_metrics.HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                path=path_label,
                status=status_label,
            ).inc()
            sie_metrics.HTTP_REQUEST_DURATION.labels(
                method=request.method,
                path=path_label,
            ).observe(elapsed)


class AppFactory:
    """Factory for creating the SIE Config Service FastAPI application."""

    @classmethod
    def create_app(cls) -> FastAPI:
        """Create and configure the FastAPI application.

        Returns:
            Configured FastAPI application instance.
        """
        app = FastAPI(
            title="SIE Config Service",
            description="Config control plane for SIE clusters",
            version="0.1.0",
            lifespan=cls._create_lifespan(),
        )

        # Prometheus HTTP middleware must wrap the app BEFORE the
        # routers mount so it observes every request (including
        # errors raised before a route matches, e.g. body-size
        # rejections). Starlette applies middleware in outer-to-inner
        # order, so adding it first means it's the outermost layer.
        app.add_middleware(_PrometheusHTTPMiddleware)

        app.include_router(health_router)
        app.include_router(config_router)
        app.include_router(metrics_router)

        return app

    @classmethod
    def _create_lifespan(cls) -> Callable[[FastAPI], Any]:
        """Create the lifespan context manager for the application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan manager."""
            logger.info("Starting SIE Config Service")

            async with (
                cls._model_registry(app),
                cls._config_store(app),
                cls._nats_publisher(app),
            ):
                yield

            logger.info("Stopped SIE Config Service")

        return lifespan

    @classmethod
    @asynccontextmanager
    async def _model_registry(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Initialize ModelRegistry for model->bundle mapping."""
        bundles_dir = Path(os.environ.get("SIE_BUNDLES_DIR", str(_DEFAULT_BUNDLES_DIR)))
        models_dir = Path(os.environ.get("SIE_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))

        try:
            model_registry = ModelRegistry(bundles_dir, models_dir)
            app.state.model_registry = model_registry
            unrouteable = model_registry.unrouteable_models
            logger.info(
                "ModelRegistry initialized: %d bundles, %d models (%d unrouteable)",
                len(model_registry.list_bundles()),
                len(model_registry.list_models()),
                len(unrouteable),
            )
            # Seed the models gauge. At this point every model came
            # from disk; the `api`-sourced tally catches up once
            # ConfigStore restore runs in `_config_store`.
            sie_metrics.update_models_gauge(
                api_count=0,
                filesystem_count=len(model_registry.list_models()),
            )
        except Exception:
            logger.exception("Failed to initialize ModelRegistry, continuing without it")
            app.state.model_registry = None

        yield

    @classmethod
    @asynccontextmanager
    async def _config_store(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Initialize config store for persisting API-added model configs."""
        config_dir = os.environ.get("SIE_CONFIG_STORE_DIR")
        if config_dir:
            store = ConfigStore(config_dir)
            app.state.config_store = store
            initial_epoch = store.read_epoch()
            logger.info("Config store initialized at %s (epoch=%d)", config_dir, initial_epoch)
            # Mirror the persisted epoch into Prometheus so the
            # `sie_config_epoch` gauge reflects reality immediately on
            # startup. Without this, dashboards would read 0 until the
            # first successful `POST /v1/configs/models` call bumped
            # the counter — which, crucially, never happens in a
            # read-only control plane.
            sie_metrics.set_epoch(initial_epoch)

            if os.environ.get("SIE_CONFIG_RESTORE", "").lower() == "true":
                model_registry: ModelRegistry | None = app.state.model_registry
                if model_registry is None:
                    logger.warning("Cannot restore configs -- ModelRegistry not initialized")
                else:
                    stored_models = store.load_all_models()
                    for model_id, model_config in stored_models.items():
                        try:
                            model_registry.add_model_config(model_config)
                            logger.info("Restored model from config store: %s", model_id)
                        except Exception:
                            logger.exception("Failed to restore model: %s", model_id)
                    if stored_models:
                        logger.info("Restored %d models from config store", len(stored_models))
                    # Recompute the split now that API-added models
                    # have been folded back into the registry.
                    api_count = len(store.list_models())
                    total = len(model_registry.list_models())
                    sie_metrics.update_models_gauge(
                        api_count=api_count,
                        filesystem_count=max(total - api_count, 0),
                    )
        else:
            app.state.config_store = None

        yield

    @classmethod
    @asynccontextmanager
    async def _nats_publisher(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Initialize NATS publisher for config distribution."""
        nats_url = os.environ.get("SIE_NATS_URL")
        nats_publisher: NatsPublisher | None = None

        if nats_url:
            nats_publisher = NatsPublisher(nats_url=nats_url)
            await nats_publisher.connect()
            app.state.nats_publisher = nats_publisher
        else:
            app.state.nats_publisher = None
            logger.info("NATS not configured (SIE_NATS_URL not set) -- config distribution disabled")

        try:
            yield
        finally:
            if nats_publisher:
                await nats_publisher.disconnect()
