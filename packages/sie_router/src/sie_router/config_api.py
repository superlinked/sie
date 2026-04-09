from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any

import orjson
import yaml
from fastapi import APIRouter, HTTPException, Request, Response

from sie_router.config_store import ConfigStore, EpochCASError
from sie_router.model_registry import BundleConflictError, ModelNotFoundError, ModelRegistry, parse_model_spec
from sie_router.nats_manager import NatsManager
from sie_router.registry import WorkerRegistry
from sie_router.types import AuditEntry

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("sie.audit")

router = APIRouter(prefix="/v1/configs", tags=["config"])

_MAX_CAS_RETRIES = 3
_MAX_CONFIG_BODY_BYTES = 1_048_576  # 1 MiB
_SERVING_READINESS_TIMEOUT_S = 3.0
_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")

# In-memory idempotency cache: maps Idempotency-Key -> (status_code, response_body, payload_hash).
# _idempotency_in_flight tracks keys currently being processed to prevent duplicate execution.
_MAX_IDEMPOTENCY_CACHE_SIZE = 1000
_idempotency_cache: OrderedDict[str, tuple[int, str, str]] = OrderedDict()
_idempotency_lock = asyncio.Lock()
_idempotency_in_flight: dict[str, asyncio.Event] = {}


def _emit_audit_log(
    request: Request,
    *,
    event: str,
    status: int,
    model: str | None = None,
    latency_ms: float | None = None,
    body_bytes: int | None = None,
) -> None:
    """Emit a structured audit log entry for config API operations."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    token_id = hashlib.sha256(token.encode()).hexdigest()[:12] if token else None

    entry = AuditEntry(
        event=event,
        method=request.method,
        endpoint=str(request.url.path),
        status=status,
        token_id=token_id,
        model=model,
        body_bytes=body_bytes,
        latency_ms=latency_ms,
    )
    audit_logger.info(orjson.dumps(entry.to_dict()).decode())


def _validate_model_id(model_id: str) -> None:
    """Validate model_id to prevent path traversal."""
    if ".." in model_id or "\\" in model_id or not _MODEL_ID_PATTERN.match(model_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_model_id", "message": "Model ID contains invalid characters."},
        )


def _check_read_auth(request: Request) -> None:
    """Validate read auth (inference token or admin token)."""
    auth_token = os.environ.get("SIE_AUTH_TOKEN")
    admin_token = os.environ.get("SIE_ADMIN_TOKEN")
    if auth_token is None and admin_token is None:
        return  # No auth configured

    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token_match = (auth_token is not None and hmac.compare_digest(token, auth_token)) or (
        admin_token is not None and hmac.compare_digest(token, admin_token)
    )
    if not token_match:
        raise HTTPException(status_code=403, detail="Invalid token")


def _check_write_auth(request: Request) -> None:
    """Validate write auth (admin token, or inference token as fallback)."""
    admin_token = os.environ.get("SIE_ADMIN_TOKEN")
    if admin_token is None:
        # If SIE_ADMIN_TOKEN is not set, refuse writes when SIE_AUTH_TOKEN
        # is present (inference token must not implicitly grant write access).
        if os.environ.get("SIE_AUTH_TOKEN"):
            raise HTTPException(
                status_code=403,
                detail="Write operations require SIE_ADMIN_TOKEN (inference token is not sufficient).",
            )
        return  # No auth configured at all

    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not hmac.compare_digest(token, admin_token):
        raise HTTPException(status_code=403, detail="Admin token required for config mutations")


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List all registered model IDs with their profiles and source."""
    _check_read_auth(request)
    model_registry: ModelRegistry = request.app.state.model_registry
    config_store: ConfigStore | None = getattr(request.app.state, "config_store", None)

    api_models = set(await asyncio.to_thread(config_store.list_models)) if config_store else set()

    models = []
    for model_name in model_registry.list_models():
        profile_names = sorted(model_registry.get_model_profile_names(model_name))
        source = "api" if model_name in api_models else "filesystem"
        models.append(
            {
                "model_id": model_name,
                "profiles": profile_names,
                "source": source,
            }
        )

    return {"models": models}


@router.get("/models/{model_id:path}")
async def get_model(request: Request, model_id: str) -> Response:
    """Return stored YAML config for a model."""
    _check_read_auth(request)
    _validate_model_id(model_id)
    config_store: ConfigStore | None = getattr(request.app.state, "config_store", None)
    model_registry: ModelRegistry = request.app.state.model_registry

    if not model_registry.model_exists(model_id):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "model_id": model_id,
                "message": f"Model '{model_id}' does not exist in the catalog.",
            },
        )

    # Try config store first (API-added), then filesystem
    content = (await asyncio.to_thread(config_store.read_model, model_id)) if config_store else None
    if content:
        return Response(content=content, media_type="application/x-yaml")

    # Model exists in registry but not in config store — it's from filesystem
    # We don't have the raw YAML easily, return a 200 with basic info
    info = model_registry.get_model_info(model_id)
    if info:
        data = {"sie_id": model_id, "source": "filesystem", "bundles": info.bundles}
        return Response(content=yaml.safe_dump(data, default_flow_style=False), media_type="application/x-yaml")

    raise HTTPException(status_code=404, detail={"error": "model_not_found", "model_id": model_id})


@router.post("/models")
async def add_model(request: Request) -> Response:
    """Add a single model config or append profiles to an existing model."""
    _check_write_auth(request)
    start_time = time.monotonic()

    model_registry: ModelRegistry = request.app.state.model_registry
    nats_manager: NatsManager | None = getattr(request.app.state, "nats_manager", None)
    config_store: ConfigStore | None = getattr(request.app.state, "config_store", None)
    worker_registry: WorkerRegistry = request.app.state.registry

    # Check NATS availability (required for config distribution)
    if nats_manager and not nats_manager.connected:
        raise HTTPException(
            status_code=503,
            detail={"error": "nats_unavailable", "message": "NATS not connected — cannot distribute config changes."},
        )

    # Parse YAML body
    body = await request.body()
    if len(body) > _MAX_CONFIG_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "payload_too_large",
                "message": f"Config body exceeds {_MAX_CONFIG_BODY_BYTES} bytes limit.",
            },
        )

    # D4: Idempotency-Key support (checked after body read for payload hash)
    idempotency_key = request.headers.get("Idempotency-Key")
    body_hash = hashlib.sha256(body).hexdigest()
    if idempotency_key:
        while True:
            async with _idempotency_lock:
                cached = _idempotency_cache.get(idempotency_key)
                if cached is not None:
                    cached_status, cached_body, cached_payload_hash = cached
                    if body_hash != cached_payload_hash:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "error": "idempotency_mismatch",
                                "message": "Idempotency-Key reused with different payload.",
                            },
                        )
                    return Response(
                        content=cached_body,
                        status_code=cached_status,
                        media_type="application/json",
                    )
                in_flight_event = _idempotency_in_flight.get(idempotency_key)
                if in_flight_event is None:
                    # We're the first — claim this key
                    _idempotency_in_flight[idempotency_key] = asyncio.Event()
                    break
            # Another request is processing this key — wait and retry
            await in_flight_event.wait()
    try:
        try:
            config = yaml.safe_load(body.decode())
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "parse_error", "message": f"Invalid YAML: {e}"},
            ) from e

        if not isinstance(config, dict):
            raise HTTPException(
                status_code=400,
                detail={"error": "parse_error", "message": "Expected YAML mapping at top level"},
            )

        # Validate model ID before mutating registry
        model_id = config.get("sie_id", "")
        if model_id:
            _validate_model_id(model_id)

        # Add to registry
        try:
            created_profiles, skipped_profiles, affected_bundles = model_registry.add_model_config(config)
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail={"error": "validation_error", "details": [{"message": str(e)}]},
            ) from e

        # D2: 409 Conflict if existing model has different content for same profiles
        if not created_profiles and skipped_profiles and config_store:
            existing_yaml = await asyncio.to_thread(config_store.read_model, model_id)
            if existing_yaml:
                try:
                    existing = yaml.safe_load(existing_yaml) or {}
                    existing_profiles = existing.get("profiles", {})
                    conflicting = []
                    for pname in skipped_profiles:
                        new_profile = config.get("profiles", {}).get(pname, {})
                        old_profile = existing_profiles.get(pname, {})
                        if new_profile != old_profile:
                            conflicting.append(pname)
                    if conflicting:
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error": "content_conflict",
                                "model_id": model_id,
                                "conflicting_profiles": conflicting,
                                "message": f"Profile(s) {conflicting} exist with different content. Config API is append-only.",
                            },
                        )
                except yaml.YAMLError:
                    pass  # If we can't parse existing, skip conflict check

        # Persist to config store (with epoch CAS)
        epoch = 0
        config_yaml = ""
        if config_store and created_profiles:
            for attempt in range(_MAX_CAS_RETRIES):
                try:
                    # Re-read and re-merge on each attempt to use fresh state
                    current_epoch = await asyncio.to_thread(config_store.read_epoch)
                    existing_yaml = await asyncio.to_thread(config_store.read_model, model_id)
                    if existing_yaml:
                        try:
                            existing_config = yaml.safe_load(existing_yaml) or {}
                        except yaml.YAMLError:
                            existing_config = {}
                        merged_profiles = existing_config.get("profiles", {})
                        merged_profiles.update(config.get("profiles", {}))
                        merged_config = dict(config)
                        merged_config["profiles"] = merged_profiles
                    else:
                        merged_config = config

                    config_yaml = yaml.dump(merged_config, default_flow_style=False, sort_keys=False)

                    # Write model config FIRST (idempotent), then bump epoch
                    await asyncio.to_thread(config_store.write_model, model_id, config_yaml)
                    epoch = await asyncio.to_thread(config_store.cas_epoch, current_epoch)
                    break
                except EpochCASError:
                    if attempt == _MAX_CAS_RETRIES - 1:
                        logger.error("Epoch CAS failed after %d retries for model %s", _MAX_CAS_RETRIES, model_id)
                        raise HTTPException(
                            status_code=503,
                            detail={"error": "cas_conflict", "message": "Concurrent config mutation — retry."},
                        ) from None
                    logger.warning("Epoch CAS retry %d for model %s", attempt + 1, model_id)
                    await asyncio.sleep(0.01)
        else:
            config_yaml = body.decode()

        # Compute bundle config hashes for affected bundles
        bundle_config_hashes = {}
        for bundle_id in affected_bundles:
            bundle_config_hashes[bundle_id] = model_registry.compute_bundle_config_hash(bundle_id)

        # Publish NATS notification
        nats_publish_failed = False
        if nats_manager and nats_manager.connected and created_profiles:
            try:
                await nats_manager.publish_config_notification(
                    model_id=model_id,
                    profiles_added=created_profiles,
                    affected_bundles=affected_bundles,
                    bundle_config_hashes=bundle_config_hashes,
                    epoch=epoch,
                    model_config_yaml=config_yaml,
                )
            except Exception:
                logger.exception("Failed to publish NATS notification for model %s", model_id)
                nats_publish_failed = True

        # Wait for serving readiness (originating router's connected workers only)
        router_id = nats_manager.router_id if nats_manager else "standalone"
        ack_result = await _wait_for_serving_readiness(
            worker_registry=worker_registry,
            affected_bundles=affected_bundles,
            bundle_config_hashes=bundle_config_hashes,
            timeout_s=_SERVING_READINESS_TIMEOUT_S,
        )

        # Build response
        warnings = list(ack_result.get("warnings", []))
        if nats_publish_failed:
            warnings.append(
                "nats_publish_failed: Config persisted but NATS notification failed. Workers may not be updated."
            )

        routable_bundles_by_profile = {}
        for profile_name in created_profiles:
            routable_bundles_by_profile[profile_name] = affected_bundles

        # 201 if any profiles were created, 200 only if all skipped
        status_code = 201 if created_profiles else 200
        response_body: dict[str, Any] = {
            "model_id": model_id,
            "created_profiles": created_profiles,
            "existing_profiles_skipped": skipped_profiles,
            "warnings": warnings,
            "routable_bundles_by_profile": routable_bundles_by_profile,
            "worker_ack_pending": ack_result["worker_ack_pending"],
            "router_id": router_id,
        }

        if created_profiles:
            response_body.update(
                {
                    "eligible_bundles_count": len(affected_bundles),
                    "eligible_bundles_with_workers_count": ack_result["eligible_bundles_with_workers_count"],
                    "acked_workers": ack_result["acked_workers"],
                    "total_eligible": ack_result["total_eligible"],
                    "pending_workers": ack_result["pending_workers"],
                }
            )

        response_json = orjson.dumps(response_body).decode()

        # D4: Cache response for idempotency
        if idempotency_key:
            async with _idempotency_lock:
                _idempotency_cache[idempotency_key] = (status_code, response_json, body_hash)
                _idempotency_cache.move_to_end(idempotency_key)
                while len(_idempotency_cache) > _MAX_IDEMPOTENCY_CACHE_SIZE:
                    _idempotency_cache.popitem(last=False)  # Evict oldest (FIFO)
                # Notify waiters and remove in-flight marker
                event = _idempotency_in_flight.pop(idempotency_key, None)
                if event is not None:
                    event.set()

    except Exception:
        # Clean up in-flight marker on any error so waiters aren't stuck forever
        if idempotency_key:
            async with _idempotency_lock:
                event = _idempotency_in_flight.pop(idempotency_key, None)
                if event is not None:
                    event.set()
        raise

    # D3: Emit audit log
    elapsed_ms = (time.monotonic() - start_time) * 1000
    _emit_audit_log(
        request,
        event="config.add_model",
        status=status_code,
        model=model_id,
        latency_ms=round(elapsed_ms, 2),
        body_bytes=len(body),
    )

    return Response(
        content=response_json,
        status_code=status_code,
        media_type="application/json",
    )


async def _wait_for_serving_readiness(
    *,
    worker_registry: WorkerRegistry,
    affected_bundles: list[str],
    bundle_config_hashes: dict[str, str],
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Wait for serving readiness on this router's connected workers.

    Serving readiness: >=1 healthy worker ACK per eligible bundle that has
    >=1 healthy connected worker on this router.

    Returns:
        Dict with ack stats.
    """
    warnings: list[str] = []

    # Count eligible workers per bundle (on this router)
    eligible_bundles_with_workers = 0
    total_eligible = 0
    for bundle_id in affected_bundles:
        workers_in_bundle = [w for w in worker_registry.healthy_workers if w.bundle.lower() == bundle_id.lower()]
        if workers_in_bundle:
            eligible_bundles_with_workers += 1
            total_eligible += len(workers_in_bundle)

    if total_eligible == 0:
        warnings.append("no_eligible_workers_connected")
        return {
            "worker_ack_pending": True,
            "eligible_bundles_with_workers_count": 0,
            "acked_workers": 0,
            "total_eligible": 0,
            "pending_workers": 0,
            "warnings": warnings,
        }

    # Poll for ACKs (bundle_config_hash match) using asyncio event-based waiting
    deadline = time.monotonic() + timeout_s
    acked_workers = 0

    while time.monotonic() < deadline:
        acked_workers = 0
        all_bundles_met = True

        for bundle_id in affected_bundles:
            expected_hash = bundle_config_hashes.get(bundle_id, "")
            workers_in_bundle = [w for w in worker_registry.healthy_workers if w.bundle.lower() == bundle_id.lower()]
            if not workers_in_bundle:
                continue

            bundle_acked = sum(1 for w in workers_in_bundle if w.bundle_config_hash == expected_hash)
            acked_workers += bundle_acked

            if bundle_acked == 0:
                all_bundles_met = False

        if all_bundles_met and acked_workers > 0:
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Use asyncio.wait_for with an event or short sleep to avoid busy-polling.
        # Sleep for min(remaining, 0.2s) to reduce CPU usage vs the original 0.1s.
        await asyncio.sleep(min(remaining, 0.2))

    pending_workers = total_eligible - acked_workers
    worker_ack_pending = pending_workers > 0

    if worker_ack_pending:
        warnings.append(
            "Serving readiness not met within timeout. Model may not be immediately servable on this router."
        )

    return {
        "worker_ack_pending": worker_ack_pending,
        "eligible_bundles_with_workers_count": eligible_bundles_with_workers,
        "acked_workers": acked_workers,
        "total_eligible": total_eligible,
        "pending_workers": pending_workers,
        "warnings": warnings,
    }


@router.post("/resolve")
async def resolve_config(request: Request) -> Response:
    """Resolve a model spec to its bundle and routing info.

    Accepts a JSON body with 'model' (required) and optional 'bundle' override.
    Returns the resolved bundle, compatible bundles, and profile names.
    """
    _check_read_auth(request)
    model_registry: ModelRegistry = request.app.state.model_registry

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "parse_error", "message": f"Invalid JSON: {e}"},
        ) from e

    model_spec = data.get("model", "")
    if not model_spec:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_field", "message": "'model' field is required"},
        )

    try:
        bundle_override, model_name = parse_model_spec(model_spec)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_model_spec", "message": str(e)},
        ) from e

    # Allow explicit bundle override from request body
    if not bundle_override:
        bundle_override = data.get("bundle")

    try:
        resolved_bundle = model_registry.resolve_bundle(model_name, bundle_override)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"error": "model_not_found", "model": model_name, "message": str(e)},
        ) from e
    except BundleConflictError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "bundle_conflict",
                "model": model_name,
                "bundle": e.bundle,
                "compatible_bundles": e.compatible_bundles,
                "message": str(e),
            },
        ) from e

    model_info = model_registry.get_model_info(model_name)
    profile_names = sorted(model_registry.get_model_profile_names(model_name))

    result: dict[str, Any] = {
        "model": model_name,
        "resolved_bundle": resolved_bundle,
        "compatible_bundles": model_info.bundles if model_info else [resolved_bundle],
        "profiles": profile_names,
    }

    return Response(
        content=orjson.dumps(result),
        status_code=200,
        media_type="application/json",
    )


@router.get("/bundles")
async def list_bundles(request: Request) -> dict[str, Any]:
    """List all known bundles with capacity info."""
    _check_read_auth(request)
    model_registry: ModelRegistry = request.app.state.model_registry
    worker_registry: WorkerRegistry = request.app.state.registry

    bundles = []
    for bundle_name in model_registry.list_bundles():
        info = model_registry.get_bundle_info(bundle_name)
        if not info:
            continue

        connected_workers = sum(1 for w in worker_registry.healthy_workers if w.bundle.lower() == bundle_name.lower())

        bundles.append(
            {
                "bundle_id": info.name,
                "priority": info.priority,
                "adapter_count": len(info.adapters),
                "source": "filesystem",
                "connected_workers": connected_workers,
            }
        )

    return {"bundles": bundles}


@router.get("/bundles/{bundle_id}")
async def get_bundle(request: Request, bundle_id: str) -> Response:
    """Return bundle metadata YAML."""
    _check_read_auth(request)
    model_registry: ModelRegistry = request.app.state.model_registry

    info = model_registry.get_bundle_info(bundle_id)
    if not info:
        raise HTTPException(
            status_code=404,
            detail={"error": "bundle_not_found", "bundle_id": bundle_id},
        )

    data = {
        "name": info.name,
        "priority": info.priority,
        "source": "filesystem",
        "adapters": info.adapters,
    }
    return Response(content=yaml.safe_dump(data, default_flow_style=False), media_type="application/x-yaml")
