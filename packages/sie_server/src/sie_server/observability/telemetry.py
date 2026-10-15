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

from sie_server.core.data_dir import ensure_data_dir, resolve_data_dir
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

_DEFAULT_ENABLED = True
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_DEFAULT_ON_NOTICE = (
    "Anonymous usage telemetry is on by default. From 0.13.0 it is off unless SIE_TELEMETRY_ENABLED=1. "
    "Set SIE_TELEMETRY_ENABLED=1 to keep sending it, or SIE_TELEMETRY_DISABLED=1 to stop now. "
    "What is sent: https://github.com/superlinked/sie#telemetry"
)

_default_on_notice_shown = False


def _warn_unrecognised(name: str) -> None:
    logger.warning(
        "%s has a value that is not recognised, so telemetry is off. Use 1, true, yes or on, or 0, false, no or off.",
        name,
    )


def _resolve_telemetry() -> tuple[bool, str]:
    disabled = os.environ.get("SIE_TELEMETRY_DISABLED", "").strip().lower()
    if disabled and disabled not in _FALSE_VALUES:
        if disabled not in _TRUE_VALUES:
            _warn_unrecognised("SIE_TELEMETRY_DISABLED")
        return False, "SIE_TELEMETRY_DISABLED"
    if os.environ.get("DO_NOT_TRACK", "").strip() == "1":
        return False, "DO_NOT_TRACK"
    enabled = os.environ.get("SIE_TELEMETRY_ENABLED", "").strip().lower()
    if not enabled:
        return _DEFAULT_ENABLED, "default"
    if enabled in _TRUE_VALUES:
        return True, "SIE_TELEMETRY_ENABLED"
    if enabled not in _FALSE_VALUES:
        _warn_unrecognised("SIE_TELEMETRY_ENABLED")
    return False, "SIE_TELEMETRY_ENABLED"


def _warn_default_on_once() -> None:
    global _default_on_notice_shown
    if _default_on_notice_shown:
        return
    _default_on_notice_shown = True
    logger.warning(_DEFAULT_ON_NOTICE)


def _get_or_create_worker_id(data_dir: str | None = None) -> str:
    if data_dir is None:
        resolved = resolve_data_dir()
        if resolved is None:
            return str(uuid.uuid4())
        data_dir = str(resolved)

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
        ensure_data_dir(Path(data_dir))
        fd = os.open(id_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(new_id)
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
    enabled, source = _resolve_telemetry()
    logger.info("telemetry: %s (%s)", "on" if enabled else "off", source)
    if not enabled:
        yield
        return
    if source == "default":
        _warn_default_on_once()

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
