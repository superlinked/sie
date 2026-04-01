import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI

from sie_server.api.encode import router as encode_router
from sie_server.api.extract import router as extract_router
from sie_server.api.health import router as health_router
from sie_server.api.metrics import router as metrics_router
from sie_server.api.models import router as models_router
from sie_server.api.openai_compat import router as openai_router
from sie_server.api.openapi import setup_custom_openapi_schema
from sie_server.api.root import router as root_router
from sie_server.api.score import router as score_router
from sie_server.api.ws import init_server_start_time
from sie_server.api.ws import router as ws_router
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.engine import EngineConfig
from sie_server.core.memory import MemoryConfig
from sie_server.core.readiness import mark_not_ready, mark_ready
from sie_server.core.registry import ModelRegistry
from sie_server.core.shutdown import ShutdownMiddleware, ShutdownState, setup_signal_handlers
from sie_server.observability.gpu import _init_nvml, shutdown_nvml
from sie_server.observability.tracing import setup_tracing

logger = logging.getLogger(__name__)


class AppFactory:
    @classmethod
    def create_app(cls, config: AppStateConfig) -> FastAPI:
        shutdown_state = ShutdownState()
        app = FastAPI(
            title="SIE Server",
            description="Search Inference Engine - GPU inference server for search workloads",
            version="0.1.0",
            lifespan=cls._create_lifespan(config, shutdown_state),
        )
        # Add graceful shutdown middleware (for spot instance preemption)
        app.add_middleware(ShutdownMiddleware, shutdown_state=shutdown_state)  # type: ignore[invalid-argument-type]

        # Setup OpenTelemetry tracing (no-op if SIE_TRACING_ENABLED is not set)
        setup_tracing(app)

        # Register routers
        app.include_router(root_router)
        app.include_router(health_router)
        app.include_router(encode_router)
        app.include_router(extract_router)
        app.include_router(score_router)
        app.include_router(models_router)
        app.include_router(metrics_router)
        app.include_router(ws_router)
        app.include_router(openai_router)  # OpenAI-compatible /v1/embeddings
        setup_custom_openapi_schema(app)

        return app

    @classmethod
    def _create_lifespan(cls, config: AppStateConfig, shutdown_state: ShutdownState) -> Callable[[FastAPI], Any]:
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan manager.

            Handles startup and shutdown events for the server.
            Initializes the model registry, starts hot reload, and cleans up on shutdown.

            Graceful shutdown:
            - On SIGTERM (spot preemption), stops accepting new requests (503)
            - Waits up to 25s for in-flight requests to complete
            - Then proceeds with normal shutdown (unload models, cleanup)
            """
            init_server_start_time()
            cls._configure_cuda_defaults()
            async with (
                cls._nvml(),
                cls._model_registry(config) as registry,
                cls._graceful_shutdown(shutdown_state),
                cls._readiness_handling(),
            ):
                app.state.registry = registry
                yield

        return lifespan

    @classmethod
    @asynccontextmanager
    async def _model_registry(cls, config: AppStateConfig) -> AsyncGenerator[ModelRegistry, None]:
        """For ModelRegistry lifecycle.

        Creates, starts, and cleanly shuts down the model registry and its background services.
        """
        engine_config = EngineConfig()
        memory_config = MemoryConfig(
            pressure_threshold=engine_config.memory_pressure_threshold_percent / 100.0,
            memory_check_interval_s=1.0,
        )

        models_dir = config.models_dir or str(engine_config.models_dir)

        registry = ModelRegistry(
            models_dir=models_dir,
            model_filter=config.model_filter,
            memory_config=memory_config,
            device=config.device,
            engine_config=engine_config,
        )
        try:
            # Start background services (memory monitor and hot reload)
            await registry.start_memory_monitor()
            await registry.start_hot_reload()
            yield registry
        finally:
            # Stop background services and unload models
            await registry.stop_memory_monitor()
            await registry.stop_hot_reload()

            logger.info("Shutting down, unloading models")
            await registry.unload_all_async()

    @classmethod
    @asynccontextmanager
    async def _nvml(cls) -> AsyncGenerator[None, None]:
        """For nvml lifecycle."""
        _init_nvml()
        try:
            yield
        finally:
            shutdown_nvml()

    @classmethod
    @asynccontextmanager
    async def _graceful_shutdown(cls, shutdown_state: ShutdownState) -> AsyncGenerator[None, None]:
        """For spot instance preemption."""
        setup_signal_handlers(shutdown_state)
        try:
            yield
        finally:
            if shutdown_state.in_flight > 0:
                logger.info("Waiting for %d in-flight requests to complete", shutdown_state.in_flight)
                await shutdown_state.wait_for_drain()

    @classmethod
    @asynccontextmanager
    async def _readiness_handling(cls) -> AsyncGenerator[None, None]:
        mark_ready()
        try:
            yield
        finally:
            mark_not_ready()

    @staticmethod
    def _configure_cuda_defaults() -> None:
        """Enable TF32 and cudnn autotuning for faster matmuls on Ampere+ GPUs.

        TF32 uses 19-bit precision for float32 matmuls — negligible accuracy
        impact for inference, but up to 3x faster on A100/L4/H100.
        cudnn.benchmark auto-tunes convolution algorithms for static input shapes.
        """
        if not torch.cuda.is_available():
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA defaults: TF32 enabled, cudnn.benchmark enabled")
