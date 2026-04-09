import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import orjson
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sie_router.api.root import router as root_router
from sie_router.app.app_state_config import AppStateConfig
from sie_router.config_api import router as config_router
from sie_router.config_store import ConfigStore
from sie_router.discovery import (
    KubernetesDiscovery,
    StaticDiscovery,
    WorkerConnectionManager,
    WorkerDiscovery,
)
from sie_router.dlq_listener import DlqListener
from sie_router.health import router as health_router
from sie_router.hot_reload import ConfigWatcher
from sie_router.jetstream_manager import JetStreamManager
from sie_router.metrics import router as metrics_router
from sie_router.model_registry import ModelRegistry
from sie_router.nats_manager import NatsManager
from sie_router.payload_store import create_payload_store
from sie_router.pools import PoolManager, PoolState
from sie_router.proxy import router as proxy_router
from sie_router.registry import WorkerRegistry
from sie_router.types import MachineProfile
from sie_router.version import ROUTER_VERSION, SERVER_VERSION_HEADER
from sie_router.work_publisher import WorkPublisher

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

        app.add_exception_handler(HTTPException, _http_exception_handler)

        # Register routers
        app.include_router(root_router)
        app.include_router(health_router)
        app.include_router(proxy_router)
        app.include_router(metrics_router)
        app.include_router(config_router)

        # Middleware stack (add_middleware wraps outermost-last):
        #
        # 1. HotPathMiddleware (innermost) — intercepts POST /v1/encode|score|extract
        #    and handles them directly, bypassing FastAPI's routing/DI/exception chain.
        #    Non-matching requests fall through to FastAPI.
        #
        # 2. _VersionHeaderMiddleware (outermost) — injects the version header on
        #    ALL responses (including hot path and FastAPI).  Pure ASGI, ~0 overhead.
        from sie_router.app.hot_path import HotPathMiddleware

        app.add_middleware(HotPathMiddleware)  # type: ignore[invalid-argument-type]  # pure ASGI middleware

        _version_header = (SERVER_VERSION_HEADER.lower().encode(), ROUTER_VERSION.encode())

        class _VersionHeaderMiddleware:
            def __init__(self, asgi_app: Any) -> None:
                self.app = asgi_app

            async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
                if scope["type"] != "http":
                    await self.app(scope, receive, send)
                    return

                async def send_with_header(message: dict) -> None:
                    if message["type"] == "http.response.start":
                        headers = list(message.get("headers", []))
                        headers.append(_version_header)
                        message = {**message, "headers": headers}
                    await send(message)

                await self.app(scope, receive, send_with_header)

        app.add_middleware(_VersionHeaderMiddleware)  # type: ignore[invalid-argument-type]  # pure ASGI middleware

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
                cls._config_store(app, config),
                cls._nats_manager(app, config),
                cls._work_publisher(app),
                cls._dlq_listener(app),
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
            data = orjson.loads(json_str)
        except orjson.JSONDecodeError as e:
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
    async def _config_store(cls, app: FastAPI, config: AppStateConfig) -> AsyncGenerator[None, None]:  # noqa: ARG003
        """Initialize config store for persisting API-added model configs."""
        config_dir = os.environ.get("SIE_CONFIG_STORE_DIR")
        if config_dir:
            store = ConfigStore(config_dir)
            app.state.config_store = store
            logger.info("Config store initialized at %s (epoch=%d)", config_dir, store.read_epoch())

            # Warn if NATS is configured but config store is local.
            # In multi-router mode this is an error; in single-router dev it's acceptable.
            nats_url = os.environ.get("SIE_NATS_URL")
            multi_router = os.environ.get("SIE_MULTI_ROUTER", "").lower() in ("1", "true")
            if nats_url and not config_dir.startswith(("s3://", "gs://")):
                msg = (
                    f"SIE_NATS_URL is set ({nats_url}) but SIE_CONFIG_STORE_DIR "
                    f"is local ({config_dir}). Multi-router deployments require "
                    f"a shared config store (s3:// or gs://) to ensure config "
                    f"consistency across routers. Either unset SIE_NATS_URL or "
                    f"set SIE_CONFIG_STORE_DIR to an S3/GCS path."
                )
                if multi_router:
                    logger.error(msg)
                    raise RuntimeError(msg)
                logger.warning(msg)

            # Restore model configs from store if enabled
            if os.environ.get("SIE_CONFIG_RESTORE", "").lower() == "true":
                model_registry: ModelRegistry | None = app.state.model_registry
                if model_registry is None:
                    logger.warning("Cannot restore configs — ModelRegistry not initialized")
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
        else:
            app.state.config_store = None

        yield

    @classmethod
    @asynccontextmanager
    async def _nats_manager(cls, app: FastAPI, config: AppStateConfig) -> AsyncGenerator[None, None]:  # noqa: ARG003
        """Initialize NATS manager for config distribution."""
        nats_url = os.environ.get("SIE_NATS_URL")
        nats_manager: NatsManager | None = None

        if nats_url:
            model_registry: ModelRegistry | None = app.state.model_registry
            if model_registry is None:
                logger.warning("NATS URL configured but ModelRegistry not initialized — skipping NATS manager setup")
                app.state.nats_manager = None
            else:

                async def on_config_notification(data: dict) -> None:
                    """Handle config notification from another router."""
                    model_config_yaml = data.get("model_config", "")
                    if model_config_yaml:
                        try:
                            config_dict = yaml.safe_load(model_config_yaml)
                            model_registry.add_model_config(config_dict)
                        except Exception:
                            logger.exception(
                                "Failed to apply config notification for model %s",
                                data.get("model_id"),
                            )

                async def on_reconnect() -> None:
                    """Reconcile from config store on NATS reconnect."""
                    config_store: ConfigStore | None = getattr(app.state, "config_store", None)
                    if config_store:
                        stored_epoch = config_store.read_epoch()
                        logger.info("NATS reconnect — checking epoch (stored=%d)", stored_epoch)
                        stored_models = config_store.load_all_models()
                        for model_id, model_config in stored_models.items():
                            try:
                                model_registry.add_model_config(model_config)
                            except Exception:  # noqa: BLE001 — best-effort reconciliation
                                logger.debug("Failed to reconcile model %s on reconnect", model_id)

                nats_manager = NatsManager(
                    nats_url=nats_url,
                    on_config_notification=on_config_notification,
                    on_reconnect=on_reconnect,
                )
                await nats_manager.connect()
                app.state.nats_manager = nats_manager
        else:
            app.state.nats_manager = None
            logger.info("NATS not configured (SIE_NATS_URL not set) — config distribution disabled")

        try:
            yield
        finally:
            if nats_manager:
                await nats_manager.disconnect()

    @classmethod
    @asynccontextmanager
    async def _work_publisher(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Initialize WorkPublisher for queue-based cluster routing."""
        work_publisher: WorkPublisher | None = None

        if os.environ.get("SIE_CLUSTER_ROUTING") != "queue":
            app.state.work_publisher = None
        else:
            nats_manager: NatsManager | None = app.state.nats_manager
            if nats_manager is None:
                logger.warning("SIE_CLUSTER_ROUTING=queue but NATS manager not initialized — skipping WorkPublisher")
                app.state.work_publisher = None
            elif nats_manager.nc is None:
                logger.warning("SIE_CLUSTER_ROUTING=queue but NATS connection is None — skipping WorkPublisher")
                app.state.work_publisher = None
            else:
                nc = nats_manager.nc
                js = nc.jetstream()
                jsm = JetStreamManager(nc=nc, js=js)
                payload_store = create_payload_store(os.environ.get("SIE_PAYLOAD_STORE_URL"))
                work_publisher = WorkPublisher(
                    nc=nc,
                    js=js,
                    jsm=jsm,
                    router_id=nats_manager.router_id,
                    payload_store=payload_store,
                )
                await work_publisher.start()
                app.state.work_publisher = work_publisher

                # Register reconnect handler to clear JSM caches and re-subscribe inbox
                nats_manager.add_reconnect_handler(work_publisher.handle_reconnect)

        try:
            yield
        finally:
            if work_publisher:
                await work_publisher.stop()

    @classmethod
    @asynccontextmanager
    async def _dlq_listener(cls, app: FastAPI) -> AsyncGenerator[None, None]:
        """Start DLQ advisory listener when queue routing is active."""
        dlq_listener: DlqListener | None = None

        work_publisher: WorkPublisher | None = app.state.work_publisher
        if work_publisher is not None:
            nats_manager: NatsManager | None = app.state.nats_manager
            if nats_manager is not None and nats_manager.nc is not None:
                nc = nats_manager.nc
                js = nc.jetstream()
                dlq_listener = DlqListener(nc=nc, js=js)
                await dlq_listener.start()
                app.state.dlq_listener = dlq_listener
            else:
                app.state.dlq_listener = None
        else:
            app.state.dlq_listener = None

        try:
            yield
        finally:
            if dlq_listener is not None:
                await dlq_listener.stop()

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
                        if pool_manager.use_kubernetes:
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
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
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
