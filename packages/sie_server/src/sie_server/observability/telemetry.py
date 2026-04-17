import asyncio
import importlib.metadata
import logging
import os
import platform
import random
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx

from sie_server.observability.gpu import get_gpu_metrics

logger = logging.getLogger(__name__)

_DEFAULT_TELEMETRY_URL = "https://telemetry.superlinked.com/api/telemetry"
_TELEMETRY_VERSION_HEADER = "X-SIE-Telemetry-Version"

# Jitter ranges (seconds)
_FIRST_UPDATE_MIN_S = 45 * 60  # 45 minutes
_FIRST_UPDATE_MAX_S = 75 * 60  # 75 minutes
_UPDATE_MIN_S = 55 * 60  # 55 minutes
_UPDATE_MAX_S = 65 * 60  # 65 minutes

_SEND_TIMEOUT_S = 10
_CONSECUTIVE_FAILURE_WARN_THRESHOLD = 3


def _is_telemetry_disabled() -> bool:
    disabled_val = os.environ.get("SIE_TELEMETRY_DISABLED", "").lower()
    if disabled_val in ("1", "true", "yes"):
        return True
    return os.environ.get("DO_NOT_TRACK") == "1"


def _get_or_create_worker_id(data_dir: str | None = None) -> str:
    if data_dir is None:
        data_dir = os.environ.get("SIE_DATA_DIR", "/tmp/sie-server")  # noqa: S108

    id_path = os.path.join(data_dir, "worker-id")

    # Try to read existing ID
    id_file = Path(id_path)

    # Try to read existing ID
    try:
        existing = id_file.read_text().strip()
        uuid.UUID(existing)  # validate
        return existing
    except (OSError, ValueError):
        pass

    # Generate new ID
    new_id = str(uuid.uuid4())

    # Try to persist
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        id_file.write_text(new_id)
    except OSError:
        logger.debug("Could not persist worker ID to %s, using ephemeral ID", id_path)

    return new_id


def _get_server_version() -> str:
    try:
        return importlib.metadata.version("sie-server")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _normalize_arch(machine: str) -> str:
    lowered = machine.lower()
    if lowered in ("x86_64", "amd64"):
        return "amd64"
    if lowered in ("aarch64", "arm64"):
        return "arm64"
    return lowered


def _build_payload(worker_id: str, event: str, gpu_names: list[str]) -> dict:
    return {
        "worker_id": worker_id,
        "sent_at": datetime.now(UTC).isoformat(),
        "event": event,
        "sie_version": _get_server_version(),
        "variant": os.environ.get("SIE_VARIANT"),
        "os": platform.system().lower(),
        "arch": _normalize_arch(platform.machine()),
        "gpus": gpu_names,
        "deployment_env": os.environ.get("SIE_DEPLOYMENT_ENV", "unknown"),
    }


async def _send_heartbeat(client: httpx.AsyncClient, url: str, payload: dict) -> bool:
    try:
        response = await client.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                _TELEMETRY_VERSION_HEADER: "1",
            },
            timeout=_SEND_TIMEOUT_S,
        )
        return response.is_success
    except Exception:  # noqa: BLE001 — telemetry must never raise
        logger.debug("Telemetry send failed", exc_info=True)
        return False


async def _telemetry_loop(
    client: httpx.AsyncClient,
    url: str,
    worker_id: str,
    gpu_names: list[str],
    stop_event: asyncio.Event,
) -> None:
    # Send INIT
    init_payload = _build_payload(worker_id, "init", gpu_names)
    await _send_heartbeat(client, url, init_payload)

    consecutive_failures = 0

    # First UPDATE delay: 45–75 minutes
    first_delay = random.uniform(_FIRST_UPDATE_MIN_S, _FIRST_UPDATE_MAX_S)  # noqa: S311
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=first_delay)
        return  # stop_event was set
    except TimeoutError:
        pass

    while not stop_event.is_set():
        payload = _build_payload(worker_id, "update", gpu_names)
        success = await _send_heartbeat(client, url, payload)

        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures == _CONSECUTIVE_FAILURE_WARN_THRESHOLD:
                logger.warning("Telemetry: %d consecutive send failures", consecutive_failures)

        # Wait 55–65 minutes before next UPDATE
        delay = random.uniform(_UPDATE_MIN_S, _UPDATE_MAX_S)  # noqa: S311
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
            return  # stop_event was set
        except TimeoutError:
            pass


@asynccontextmanager
async def telemetry_sender() -> AsyncGenerator[None, None]:
    if _is_telemetry_disabled():
        logger.info("telemetry: disabled via env var")
        yield
        return

    url = os.environ.get("SIE_TELEMETRY_URL", _DEFAULT_TELEMETRY_URL)
    worker_id = _get_or_create_worker_id()

    # Detect GPU names once (NVML must be initialized before this point)
    try:
        gpu_names = [g["name"] for g in get_gpu_metrics()]
    except Exception:  # noqa: BLE001 — telemetry must never raise
        logger.debug("Telemetry: could not detect GPUs", exc_info=True)
        gpu_names = []

    client = httpx.AsyncClient()
    stop_event = asyncio.Event()
    task: asyncio.Task[None] | None = None

    try:
        task = asyncio.create_task(_telemetry_loop(client, url, worker_id, gpu_names, stop_event))
        yield
    finally:
        stop_event.set()

        # Send TERMINATE with short timeout
        try:
            terminate_payload = _build_payload(worker_id, "terminate", gpu_names)
            await asyncio.wait_for(
                _send_heartbeat(client, url, terminate_payload),
                timeout=_SEND_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 — telemetry must never block shutdown
            logger.debug("Telemetry: TERMINATE send failed", exc_info=True)

        # Cancel the background task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await client.aclose()
