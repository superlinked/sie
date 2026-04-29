import logging
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import torch
import yaml
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
from sie_server.config.model import ModelConfig, ProfileConfig, Tasks
from sie_server.core.memory import MemoryConfig
from sie_server.core.readiness import mark_not_ready, mark_ready
from sie_server.core.registry import ModelRegistry
from sie_server.core.shutdown import ShutdownMiddleware, ShutdownState, setup_signal_handlers
from sie_server.nats_pull_loop import NatsPullLoop
from sie_server.nats_subscriber import NatsSubscriber
from sie_server.observability.gpu import _init_nvml, shutdown_nvml
from sie_server.observability.telemetry import telemetry_sender
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
        app.add_middleware(ShutdownMiddleware, shutdown_state=shutdown_state)

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
                telemetry_sender(),
                cls._model_registry(config) as registry,
                cls._nats_subscriber(registry) as nats_sub,
                cls._nats_pull_loop(registry, nats_sub) as pull_loop,
                cls._graceful_shutdown(shutdown_state),
                cls._readiness_handling(),
            ):
                app.state.registry = registry
                app.state.nats_subscriber = nats_sub
                app.state.nats_pull_loop = pull_loop
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
            # Start background services (memory monitor, idle evictor, hot reload).
            # The idle evictor is a no-op when ``idle_evict_s`` is None.
            await registry.start_memory_monitor()
            await registry.start_idle_evictor()
            await registry.start_hot_reload()

            # Preload models (shifts weight download from first-request to startup)
            await cls._preload_models(registry, config)

            yield registry
        finally:
            # Stop background services and unload models
            await registry.stop_memory_monitor()
            await registry.stop_idle_evictor()
            await registry.stop_hot_reload()

            logger.info("Shutting down, unloading models")
            await registry.unload_all_async()

    @classmethod
    @asynccontextmanager
    async def _nats_subscriber(cls, registry: ModelRegistry) -> AsyncGenerator[NatsSubscriber | None, None]:
        """For optional NATS subscriber lifecycle.

        Only starts if SIE_NATS_URL is set. Connection failures are non-fatal —
        the server starts normally without NATS.
        """
        nats_url = os.environ.get("SIE_NATS_URL")
        if not nats_url:
            yield None
            return

        async def on_model_config(model_id: str, config_yaml: str) -> None:
            try:
                raw = yaml.safe_load(config_yaml)
                if not isinstance(raw, dict):
                    logger.warning("NATS config for '%s' is not a dict, ignoring", model_id)
                    return

                # Try full ModelConfig parsing (works for complete configs with tasks, hf_id, etc.)
                try:
                    model_config = ModelConfig(**raw)
                    registry.add_config(model_config)
                    logger.info("Added full model config from NATS: %s", model_config.sie_id)
                    return
                except Exception:  # noqa: BLE001 — fallback to lightweight path
                    logger.debug("Full ModelConfig parse failed for '%s', trying minimal path", model_id)

                # Fallback: minimal config from Config API (sie_id + profiles only).
                # Create a stub ModelConfig with required fields filled in from the profile.
                sie_id = raw.get("sie_id", model_id)
                profiles_raw = raw.get("profiles", {})
                if not profiles_raw:
                    logger.warning("NATS config for '%s' has no profiles, ignoring", model_id)
                    return

                # Fallback to sie_id as hf_id (satisfies ModelConfig validator)

                # Build a minimal valid ModelConfig
                profiles = {}
                for pname, pdata in profiles_raw.items():
                    profiles[pname] = ProfileConfig(
                        adapter_path=pdata.get("adapter_path"),
                        max_batch_tokens=pdata.get("max_batch_tokens"),
                        extends=pdata.get("extends"),
                    )

                # Build a stub ModelConfig for profile/hash tracking.
                # This config may not be loadable — it's used to register the model
                # in the registry so the config hash updates and routing can reference it.
                # Actual model loading will use the full config pushed by the adapter.
                model_config = ModelConfig(
                    sie_id=sie_id,
                    hf_id=raw.get("hf_id", sie_id),
                    tasks=Tasks(),
                    profiles=profiles,
                )
                registry.add_config(model_config)
                logger.info("Added minimal model config from NATS: %s", sie_id)

            except Exception:
                logger.exception("Failed to apply NATS config for model '%s'", model_id)

        subscriber = NatsSubscriber(nats_url=nats_url, on_model_config=on_model_config)
        await subscriber.start()
        try:
            yield subscriber
        finally:
            await subscriber.stop()

    @classmethod
    @asynccontextmanager
    async def _nats_pull_loop(
        cls, registry: ModelRegistry, nats_subscriber: NatsSubscriber | None
    ) -> AsyncGenerator[NatsPullLoop | None, None]:
        """For optional NATS pull loop lifecycle.

        Only starts when SIE_CLUSTER_ROUTING=queue and a NATS connection exists.
        """
        if os.environ.get("SIE_CLUSTER_ROUTING") != "queue":
            yield None
            return

        if nats_subscriber is None:
            raise RuntimeError("SIE_CLUSTER_ROUTING=queue but no NATS subscriber available — cannot start pull loop")

        nc = nats_subscriber.nc
        if nc is None:
            raise RuntimeError("SIE_CLUSTER_ROUTING=queue but NATS connection is None — cannot start pull loop")
        js = nc.jetstream()
        bundle_id = os.environ.get("SIE_BUNDLE", "default")
        pool_name = os.environ.get("SIE_POOL", "_default")
        payload_store_url = os.environ.get("SIE_PAYLOAD_STORE_URL")

        pull_loop = NatsPullLoop(
            nc=nc,
            js=js,
            registry=registry,
            bundle_id=bundle_id,
            pool_name=pool_name,
            payload_store_url=payload_store_url,
        )
        await pull_loop.start()

        # Register pull loop reconnect handler via NatsSubscriber's public API.
        # This is invoked by the NATS client's reconnected_cb chain, ensuring
        # pull subscriptions are re-created after NATS reconnect.
        nats_subscriber.add_reconnect_handler(pull_loop.handle_reconnect)

        try:
            yield pull_loop
        finally:
            await pull_loop.stop()

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

    @classmethod
    async def _preload_models(cls, registry: ModelRegistry, config: AppStateConfig) -> None:
        """Preload models at startup (non-fatal on failure).

        Runs inside _model_registry() which enters before _readiness_handling()
        in the async-with stack, so the pod stays NotReady during preload.
        """
        if not config.preload_models:
            return

        logger.info("Preloading %d model(s): %s", len(config.preload_models), ", ".join(config.preload_models))

        # Sequential loading is intentional: parallel loads risk OOM on GPU workers
        # where VRAM is limited. For CPU workers with many models, this is slightly
        # slower but safe. Parallel preloading can be added later if needed.
        succeeded = 0
        for name in config.preload_models:
            try:
                await registry.load_async(name, config.device)
                succeeded += 1
                logger.info("Preloaded model '%s'", name)
            except Exception:
                logger.exception(
                    "Failed to preload model '%s', skipping (will lazy-load on request). "
                    "Check that the model name matches a config in the models directory.",
                    name,
                )

        logger.info("Preload complete: %d/%d models loaded", succeeded, len(config.preload_models))

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
