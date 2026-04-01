import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sie_router.api.root import router as root_router
from sie_router.app.app_state_config import AppStateConfig
from sie_router.discovery import (
    KubernetesDiscovery,
    StaticDiscovery,
    WorkerConnectionManager,
    WorkerDiscovery,
)
from sie_router.health import router as health_router
from sie_router.hot_reload import ConfigWatcher
from sie_router.metrics import router as metrics_router
from sie_router.model_registry import ModelRegistry
from sie_router.pools import PoolManager, PoolState
from sie_router.proxy import router as proxy_router
from sie_router.registry import WorkerRegistry
from sie_router.types import MachineProfile
from sie_router.version import ROUTER_VERSION, SERVER_VERSION_HEADER

logger = logging.getLogger(__name__)


async def _http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return JSON error with the server version header."""
    http_exc = exc if isinstance(exc, HTTPException) else HTTPException(status_code=500)
    return JSONResponse(
        status_code=http_exc.status_code,
        content={"detail": http_exc.detail},
        headers={SERVER_VERSION_HEADER: ROUTER_VERSION},
    )


# Default paths for bundle and model configs (relative to project root)
_DEFAULT_BUNDLES_DIR = Path(__file__).parent.parent.parent.parent.parent / "sie_server" / "bundles"
_DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "sie_server" / "models"


class AppFactory:
    """Factory for creating the SIE Router FastAPI application."""

    @classmethod
    def create_app(cls, config: AppStateConfig) -> FastAPI:
        """Create and configure the FastAPI application.

        Args:
            config: Application state configuration.

        Returns:
            Configured FastAPI application instance.
        """
        app = FastAPI(
            title="SIE Router",
            description="Stateless request router for SIE elastic cloud deployments",
            version="0.1.0",
            lifespan=cls._create_lifespan(config),
        )

        # Version header middleware (runs on every response including errors)
        @app.middleware("http")
        async def _add_version_header(
            request: Request,
            call_next: Callable[[Request], Any],
        ) -> JSONResponse:
            response = await call_next(request)
            response.headers[SERVER_VERSION_HEADER] = ROUTER_VERSION
            return response

        app.add_exception_handler(HTTPException, _http_exception_handler)

        # Register routers
        app.include_router(root_router)
        app.include_router(health_router)
        app.include_router(proxy_router)
        app.include_router(metrics_router)

        return app

    @classmethod
    def _create_lifespan(cls, config: AppStateConfig) -> Callable[[FastAPI], Any]:
        """Create the lifespan context manager for the application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan manager."""
            logger.info("Starting SIE Router")

            async with (
                cls._machine_profiles(app),
                cls._model_registry(app, config),
                cls._pool_manager(app, config) as pool_manager,
                cls._worker_registry(app, pool_manager),
                cls._http_client(app),
                cls._connection_manager(app, config),
            ):
                yield

            logger.info("Stopped SIE Router")

        return lifespan

    @classmethod
    @asynccontextmanager
    async def _machine_profiles(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Load machine profiles from environment variable."""
        machine_profiles_json = os.environ.get("SIE_ROUTER_MACHINE_PROFILES", "")
        machine_profiles = cls._load_machine_profiles(machine_profiles_json)
        app.state.machine_profiles = machine_profiles
        if machine_profiles:
            logger.info(
                "Loaded %d machine profiles: %s",
                len(machine_profiles),
                list(machine_profiles.keys()),
            )
        yield

    @classmethod
    def _load_machine_profiles(cls, json_str: str) -> dict[str, MachineProfile]:
        """Load machine profiles from JSON string.

        Args:
            json_str: JSON string with machine profile definitions.
                Example: '{"l4": {"gpuType": "nvidia-l4", "spot": false}, ...}'

        Returns:
            Dict mapping profile name to MachineProfile.
        """
        if not json_str.strip():
            return {}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse SIE_ROUTER_MACHINE_PROFILES: %s", e)
            return {}

        profiles: dict[str, MachineProfile] = {}
        for name, profile_config in data.items():
            if not isinstance(profile_config, dict):
                logger.warning("Skipping invalid machine profile '%s': not a dict", name)
                continue

            profiles[name] = MachineProfile(
                name=name,
                gpu_type=profile_config.get("gpuType", ""),
                machine_type=profile_config.get("machineType", ""),
                spot=profile_config.get("spot", False),
            )

        return profiles

    @classmethod
    @asynccontextmanager
    async def _model_registry(cls, app: FastAPI, config: AppStateConfig) -> AsyncGenerator[None, None]:
        """Initialize ModelRegistry for model→bundle mapping."""
        bundles_dir = Path(os.environ.get("SIE_BUNDLES_DIR", str(_DEFAULT_BUNDLES_DIR)))
        models_dir = Path(os.environ.get("SIE_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))

        config_watcher: ConfigWatcher | None = None
        try:
            model_registry = ModelRegistry(bundles_dir, models_dir)
            app.state.model_registry = model_registry
            logger.info(
                "ModelRegistry initialized: %d bundles, %d models",
                len(model_registry.list_bundles()),
                len(model_registry.list_models()),
            )

            # Start config watcher for hot reload
            if config.enable_hot_reload:
                config_watcher = ConfigWatcher(model_registry, bundles_dir, models_dir)
                config_watcher.start()
                app.state.config_watcher = config_watcher
        except Exception:
            logger.exception("Failed to initialize ModelRegistry, continuing without it")
            app.state.model_registry = None

        try:
            yield
        finally:
            if config_watcher:
                config_watcher.stop()

    @classmethod
    @asynccontextmanager
    async def _pool_manager(cls, app: FastAPI, config: AppStateConfig) -> AsyncGenerator[PoolManager | None, None]:
        """Initialize pool manager if enabled."""
        pool_manager: PoolManager | None = None

        if config.enable_pools or config.use_kubernetes:
            logger.info("Enabling pool management (namespace=%s)", config.k8s_namespace)
            pool_manager = PoolManager(
                use_kubernetes=config.use_kubernetes,
                k8s_namespace=config.k8s_namespace,
                machine_profiles=app.state.machine_profiles,
            )
            await pool_manager.start()
            # Create the default pool with all machine profiles
            # This enables bare gpu="l4-spot" routing without explicit pool creation
            await pool_manager.create_default_pool()

        app.state.pool_manager = pool_manager
        try:
            yield pool_manager
        finally:
            if pool_manager:
                await pool_manager.stop()

    @classmethod
    @asynccontextmanager
    async def _worker_registry(cls, app: FastAPI, pool_manager: PoolManager | None) -> AsyncGenerator[None, None]:
        """Initialize worker registry with pool assignment callback."""

        async def on_worker_healthy(worker: Any) -> None:
            """Callback for when a worker becomes healthy - triggers pool assignment."""
            if not pool_manager:
                return

            # Check all pending pools and try to assign workers
            for pool in pool_manager.pools.values():
                if pool.status.state != PoolState.PENDING:
                    continue

                try:
                    # Get available workers (not already assigned to other pools)
                    registry = app.state.registry
                    assigned_urls = pool_manager.get_assigned_worker_urls()
                    available = [
                        (w.name, w.url, w.machine_profile, w.bundle)
                        for w in registry.healthy_workers
                        if w.url not in assigned_urls
                    ]

                    if pool_manager.assign_workers(pool, available):
                        logger.info(
                            "Assigned workers to pool '%s' (triggered by worker %s becoming healthy)",
                            pool.name,
                            worker.name,
                        )
                        # Update pool in K8s if using K8s mode
                        if pool_manager._use_kubernetes:
                            await pool_manager._update_pool_in_k8s(pool)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 — K8s pool processing resilience
                    logger.error(
                        "Failed to process pool '%s' for worker '%s': %s",
                        pool.name,
                        worker.name,
                        e,
                    )

        app.state.registry = WorkerRegistry(on_worker_healthy=on_worker_healthy)
        yield

    @classmethod
    @asynccontextmanager
    async def _http_client(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Create shared httpx client for connection pooling to workers."""
        client = httpx.AsyncClient()
        app.state.http_client = client
        try:
            yield
        finally:
            await client.aclose()

    @classmethod
    @asynccontextmanager
    async def _connection_manager(cls, app: FastAPI, config: AppStateConfig) -> AsyncGenerator[None, None]:
        """Initialize and manage worker discovery and connection manager."""
        discovery: WorkerDiscovery
        if config.use_kubernetes:
            logger.info(
                "Using Kubernetes discovery: %s/%s",
                config.k8s_namespace,
                config.k8s_service,
            )
            discovery = KubernetesDiscovery(
                namespace=config.k8s_namespace,
                service_name=config.k8s_service,
                port=config.k8s_port,
            )
        else:
            logger.info("Using static discovery: %d workers", len(config.worker_urls))
            discovery = StaticDiscovery(config.worker_urls)

        connection_manager = WorkerConnectionManager(
            registry=app.state.registry,
            discovery=discovery,
        )
        app.state.connection_manager = connection_manager
        await connection_manager.start()

        try:
            yield
        finally:
            await connection_manager.stop()
