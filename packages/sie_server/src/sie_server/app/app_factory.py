import logging
import os
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI

from sie_server.api.encode import router as encode_router
from sie_server.api.extract import router as extract_router
from sie_server.api.generate import router as generate_router
from sie_server.api.health import router as health_router
from sie_server.api.models import router as models_router
from sie_server.api.openai_audio import router as openai_audio_router
from sie_server.api.openai_compat import router as openai_router
from sie_server.api.openai_local import router as openai_local_router
from sie_server.api.openapi import setup_custom_openapi_schema
from sie_server.api.root import router as root_router
from sie_server.api.score import router as score_router
from sie_server.api.ws import init_server_start_time
from sie_server.api.ws import router as ws_router
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.engine import EngineConfig
from sie_server.core.memory import MemoryConfig
from sie_server.core.readiness import mark_not_ready, mark_ready, register_liveness_probe
from sie_server.core.registry import ModelRegistry
from sie_server.core.shutdown import ShutdownMiddleware, ShutdownState, setup_signal_handlers
from sie_server.ipc_server import IpcServer
from sie_server.observability.gpu import _init_nvml, shutdown_nvml
from sie_server.observability.telemetry import telemetry_sender
from sie_server.observability.tracing import setup_tracing, shutdown_tracing
from sie_server.observability.worker_telemetry import (
    configure_worker_metric_context,
    lane_from_environment,
    setup_worker_telemetry,
    shutdown_worker_telemetry,
)
from sie_server.queue_executor import QueueExecutor

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


@asynccontextmanager
async def _timed_stage(name: str, cm: Any) -> AsyncGenerator[Any, None]:
    """Wrap an async context manager to log its entry-phase elapsed time.

    Emits one structured line per stage on entry — `lifespan.stage <name>
    elapsed_s=<x>` — so cold-start tooling (issue #816) can attribute the ~5s
    `engine_boot_s` consistently seen across LTFR runs to specific lifespan
    stages (NVML init, NATS connect, telemetry handshake, etc).

    The exit-phase teardown is intentionally not timed — only the setup phase
    contributes to `engine_boot_s`. ``cm`` is typed as ``Any`` because the
    contextlib ``@asynccontextmanager`` wrapper produces a context manager
    type that ty struggles to bind through a TypeVar parameter.
    """
    t0 = time.perf_counter()
    async with cm as value:
        elapsed = time.perf_counter() - t0
        logger.info("lifespan.stage %s elapsed_s=%.3f", name, elapsed)
        yield value


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

        # Setup OpenTelemetry tracing (no-op unless the flag and endpoint are set)
        setup_tracing(app)
        setup_worker_telemetry()

        # Queue is the only supported routing mode: the worker-sidecar
        # drives NATS and talks to Python over UDS IPC. We still mount
        # the HTTP routers for /healthz and /readyz (K8s probes), the
        # landing page, and /ws/status — which is
        # the gateway's worker-registration channel: the gateway dials
        # `ws://<pod>/ws/status` to learn pool_name, bundle,
        # machine_profile, and readiness. The inference endpoints
        # (/encode, /score, /extract, /generate, /models, /v1/embeddings) are
        # historically reachable on the pod's HTTP port but are not a
        # production ingress — the Rust gateway is queue-only and
        # publishes to JetStream, not HTTP. They stay mounted so local
        # SIEClient-based checks and the existing test surface keep
        # working; no traffic reaches them in a real cluster.
        app.include_router(root_router)
        app.include_router(health_router)
        app.include_router(ws_router)
        app.include_router(encode_router)
        app.include_router(extract_router)
        app.include_router(generate_router)
        app.include_router(score_router)
        app.include_router(models_router)
        app.include_router(openai_audio_router)  # OpenAI-compatible /v1/audio/transcriptions
        app.include_router(openai_router)  # OpenAI-compatible /v1/embeddings
        # Local-only convenience (single-node/dev); intentionally NOT in the published
        # openapi.json — see cli.openapi_export. The Rust gateway is the prod API authority.
        app.include_router(openai_local_router)  # local /v1/chat/completions + /v1/rerank
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
            cls._configure_torch_threads()
            cls._configure_cuda_defaults()
            async with (
                # First to enter → last to tear down, so spans emitted during
                # other stages' shutdown (model unload, drain) are still flushed.
                _timed_stage("tracing", cls._tracing()),
                _timed_stage("metrics", cls._metrics()),
                _timed_stage("nvml", cls._nvml()),
                _timed_stage("telemetry", telemetry_sender()),
                _timed_stage("model_registry", cls._model_registry(config)) as registry,
                _timed_stage("ipc_server", cls._ipc_server(registry)) as ipc_server,
                _timed_stage("graceful_shutdown", cls._graceful_shutdown(shutdown_state)),
                _timed_stage("readiness", cls._readiness_handling()),
            ):
                app.state.registry = registry
                app.state.ipc_server = ipc_server
                # /ws/status reads queue_runtime for the canonical worker_id.
                # The Python NATS pull loop is gone, so the IPC server is the
                # remaining Python object that mirrors the sidecar identity.
                app.state.queue_runtime = ipc_server
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
            devices=config.devices,
            engine_config=engine_config,
            pool_name=config.pool_name,
            pinned_models=config.pinned_models,
        )
        configure_worker_metric_context(
            lane=lane_from_environment(),
            configs=registry.get_configs_snapshot(),
        )
        try:
            # Start background services (memory monitor, idle evictor, hot reload).
            # The idle evictor is a no-op when ``idle_evict_s`` is None.
            await registry.start_memory_monitor()
            await registry.start_idle_evictor()
            await registry.start_hot_reload()

            # Preload models (shifts weight download from first-request to startup)
            await cls._preload_models(registry, config)
            # Eager-load pinned models (models already loaded by preload are skipped)
            await cls._load_pinned_models(registry, config)

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
    async def _ipc_server(cls, registry: ModelRegistry) -> AsyncGenerator[IpcServer, None]:
        """IPC server lifecycle for the sidecar path.

        The worker-sidecar drives NATS and talks to Python over a
        UDS msgpack RPC. This context manager binds the socket, exposes
        the :class:`QueueExecutor`, and can wire the sidecar heartbeat into
        ``/readyz`` so a dead sidecar flips the pod unready.

        The IPC server starts unconditionally so the same image can serve
        direct HTTP and sidecar-driven queue deployments. Readiness only
        depends on the Rust heartbeat when ``SIE_IPC_REQUIRE_HEARTBEAT`` is
        truthy; Helm sets that env var when the worker-sidecar is present.
        """
        socket_path = os.environ.get("SIE_IPC_SOCKET_PATH", "/tmp/sie-ipc.sock")  # noqa: S108
        stale_after_ms = float(os.environ.get("SIE_IPC_STALE_AFTER_MS", "10000"))
        worker_id = os.environ.get("SIE_WORKER_ID") or os.environ.get("HOSTNAME", "worker-unknown")
        require_heartbeat = _env_flag("SIE_IPC_REQUIRE_HEARTBEAT")

        executor = QueueExecutor(registry)
        server = IpcServer(
            socket_path,
            executor,
            worker_id=worker_id,
            stale_after_ms=stale_after_ms,
        )
        await server.start()
        if require_heartbeat:
            register_liveness_probe(server.is_heartbeat_fresh)
        else:
            register_liveness_probe(None)
        try:
            yield server
        finally:
            register_liveness_probe(None)
            await server.stop()

    @classmethod
    @asynccontextmanager
    async def _tracing(cls) -> AsyncGenerator[None, None]:
        """Bounded tracing flush on shutdown.

        Setup happens in ``build_app`` (``setup_tracing``); this stage only owns
        the deterministic, bounded shutdown so exit can't stall on an unreachable
        OTLP collector.
        """
        try:
            yield
        finally:
            shutdown_tracing()

    @classmethod
    @asynccontextmanager
    async def _metrics(cls) -> AsyncGenerator[None, None]:
        """Flush the owned OTLP metrics provider after engine shutdown."""
        try:
            yield
        finally:
            shutdown_worker_telemetry()

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
            logger.info("lifespan.stage preload_models elapsed_s=0.000")
            return

        t0 = time.perf_counter()
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

        elapsed = time.perf_counter() - t0
        logger.info("Preload complete: %d/%d models loaded", succeeded, len(config.preload_models))
        logger.info("lifespan.stage preload_models elapsed_s=%.3f", elapsed)

    @classmethod
    async def _load_pinned_models(cls, registry: ModelRegistry, config: AppStateConfig) -> None:
        """Eager-load pinned models at startup (non-fatal on failure).

        Models already loaded by ``_preload_models`` are skipped so a model
        in both lists is loaded exactly once. Runs inside _model_registry()
        which enters before _readiness_handling(), keeping the pod NotReady
        until pinned models are resident.
        """
        if not config.pinned_models:
            logger.info("lifespan.stage load_pinned_models elapsed_s=0.000")
            return

        t0 = time.perf_counter()
        logger.info("Loading %d pinned model(s): %s", len(config.pinned_models), ", ".join(config.pinned_models))

        succeeded = 0
        for raw in config.pinned_models:
            # Pinned ids may be profile-qualified (``sie_id:profile``) or
            # differently-cased; resolve to the config's bare sie_id so the
            # eager-load matches the same set the eviction guard protects.
            name = registry.resolve_model_id(raw)
            if name is None:
                logger.warning(
                    "Pinned model '%s' has no matching config on this worker, skipping eager-load "
                    "(it stays eviction-protected if it loads later). "
                    "Check that the model name matches a config in the models directory.",
                    raw,
                )
                continue
            if registry.is_loaded(name):
                logger.info("Pinned model '%s' already loaded (skipping duplicate load)", name)
                succeeded += 1
                continue
            try:
                await registry.load_async(name, config.device)
                succeeded += 1
                logger.info("Pinned model '%s' loaded", name)
            except Exception:
                logger.exception(
                    "Failed to load pinned model '%s', skipping (it will lazy-load on "
                    "request and stay eviction-protected once resident). "
                    "Check that the model name matches a config in the models directory.",
                    name,
                )

        elapsed = time.perf_counter() - t0
        logger.info("Pinned model load complete: %d/%d models loaded", succeeded, len(config.pinned_models))
        logger.info("lifespan.stage load_pinned_models elapsed_s=%.3f", elapsed)

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

    @staticmethod
    def _configure_torch_threads() -> None:
        # Cap torch BLAS threads so concurrent CPU consumers (Docling per-batch
        # pool, image preprocessor pool) don't oversubscribe cores. Override
        # via SIE_TORCH_NUM_THREADS; default = half the logical cores.
        override = os.environ.get("SIE_TORCH_NUM_THREADS")
        if override is not None:
            try:
                n = int(override)
                if n < 1:
                    raise ValueError
            except ValueError:
                logger.warning(
                    "SIE_TORCH_NUM_THREADS=%r is not a positive integer; using default",
                    override,
                )
                n = max(1, (os.cpu_count() or 4) // 2)
                logger.info("torch threads: %d (default after invalid override; cpu_count=%s)", n, os.cpu_count())
            else:
                logger.info("torch threads: %d (from SIE_TORCH_NUM_THREADS)", n)
        else:
            n = max(1, (os.cpu_count() or 4) // 2)
            logger.info("torch threads: %d (default; cpu_count=%s)", n, os.cpu_count())

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already initialised; safe to ignore — only the first call has effect.
            logger.warning("torch.set_num_interop_threads(1) ignored: parallel runtime already started")
