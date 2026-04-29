from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

import orjson
import yaml
from fastapi import APIRouter, HTTPException, Request, Response

from sie_config import metrics as sie_metrics
from sie_config.config_store import ConfigStore
from sie_config.model_registry import BundleConflictError, ModelNotFoundError, ModelRegistry, parse_model_spec
from sie_config.nats_publisher import NatsPublisher, PartialPublishError
from sie_config.types import AuditEntry

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("sie.audit")

router = APIRouter(prefix="/v1/configs", tags=["config"])

_MAX_CONFIG_BODY_BYTES = 1_048_576  # 1 MiB
_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")

# In-memory idempotency cache: maps Idempotency-Key -> (status_code, response_body, payload_hash).
# The cache, its guarding lock, and the in-flight event map are all held
# on `app.state` (see `_get_idempotency_state`). They MUST NOT live at
# module scope for the same reason `_WRITE_LOCK` doesn't: `asyncio.Lock`
# and `asyncio.Event` bind to the first event loop that awaits them, so a
# module-level instance would raise
# `RuntimeError: ... is bound to a different event loop` the moment any
# second loop (test harness, subinterpreter, etc.) touches it.
_MAX_IDEMPOTENCY_CACHE_SIZE = 1000
_APP_STATE_IDEMPOTENCY_ATTR = "_config_idempotency_state"
_APP_STATE_IDEMPOTENCY_LOOP_ATTR = "_config_idempotency_state_loop"


class _IdempotencyState:
    """Per-app container for idempotency state.

    Lifetime:
    -   `cache` survives across event loops. It's a plain dict holding
        completed response bytes — no loop-bound primitives in it — so
        carrying it forward is what lets `TestClient` (which spins up a
        fresh `asyncio` loop per request) still observe prior results.
    -   `lock` and every `asyncio.Event` in `in_flight` are loop-bound
        and must be rebuilt any time the running loop changes. We do
        this lazily in `_get_idempotency_state`: when the running loop
        doesn't match `_loop`, we swap `lock` for a fresh one and drop
        `in_flight` (any waiters on the old loop are gone anyway). The
        cache is left intact.
    """

    __slots__ = ("_loop", "cache", "in_flight", "lock")

    def __init__(self) -> None:
        self.cache: OrderedDict[str, tuple[int, str, str]] = OrderedDict()
        self.lock = asyncio.Lock()
        self.in_flight: dict[str, asyncio.Event] = {}
        self._loop = asyncio.get_running_loop()

    def rebind_to_current_loop(self) -> None:
        """Rebuild loop-bound primitives against the current event loop.

        Called from `_get_idempotency_state` when the running loop has
        changed since the last time we touched the state. `cache` is
        intentionally preserved — its contents are loop-agnostic response
        bytes. `in_flight` is dropped because every Event in it is
        pinned to a dead loop; anything waiting on them has gone with
        the loop.
        """
        self.lock = asyncio.Lock()
        self.in_flight = {}
        self._loop = asyncio.get_running_loop()

    def clear(self) -> None:
        self.cache.clear()
        self.in_flight.clear()


# Per-app write lock is stored on `app.state._config_write_lock` and
# lazily created on first use. We cannot use a module-level
# `asyncio.Lock()` here because `asyncio.Lock` binds to the running
# event loop the first time it's awaited; a module-level instance
# would therefore break any test that runs multiple
# `asyncio.run(...)` iterations against the same app, and would also
# fall apart if the process ever hosts more than one event loop
# (subinterpreters, threaded test harnesses, etc.). Storing the lock
# on `app.state` scopes its lifetime to the app and its event loop.
#
# Why serialize this critical section at all:
#   - `ConfigStore.increment_epoch` is a naive read-modify-write on
#     the epoch file. Concurrent writers would both read N, both
#     write N+1, and we'd lose an epoch bump (and the NATS monotonic
#     sequence number the gateway poller relies on).
#   - Disk persist must happen before the in-memory registry mutation
#     so a restart cannot observe a registry-only model that has no
#     backing YAML. That invariant only holds if nothing slips
#     between persist and mutate.
#   - NATS publish is part of the same "one write = one delta"
#     contract, so concurrent writes must serialize their publishes
#     to keep the epoch ordering on the wire monotonic per bundle.
#   - `GET /v1/configs/export` also takes this lock so its
#     `(epoch, models)` pair is a real serialization point; without
#     that, a gateway bootstrapping mid-write could receive an epoch
#     that is ahead of the models list and silently wedge on the
#     poller's `remote == local` "in sync" branch.
#
# All read endpoints (`/models`, `/bundles`, `/epoch`, ...) remain
# fully concurrent.
_APP_STATE_WRITE_LOCK_ATTR = "_config_write_lock"
_APP_STATE_WRITE_LOOP_ATTR = "_config_write_lock_loop"


def _get_write_lock(app_state: Any) -> asyncio.Lock:
    """Lazily fetch-or-create the per-app write lock, event-loop-bound.

    `asyncio.Lock()` in Python 3.10+ binds to whichever event loop is
    running when the lock is first awaited. That's fine in production
    (uvicorn owns a single long-lived loop), but it means an app
    instance that is reused across multiple `asyncio.run(...)` calls
    (e.g. a test module that spins up fresh loops per test) would
    inherit a lock bound to a dead loop and raise
    `RuntimeError: ... is bound to a different event loop`.

    Fix: store the lock on `app.state` keyed by the current running
    loop. If the running loop doesn't match the loop the lock was
    created on, rebuild — safe because per-loop execution is
    non-overlapping by construction.
    """
    running_loop = asyncio.get_running_loop()
    existing_lock = getattr(app_state, _APP_STATE_WRITE_LOCK_ATTR, None)
    existing_loop = getattr(app_state, _APP_STATE_WRITE_LOOP_ATTR, None)
    if existing_lock is None or existing_loop is not running_loop:
        new_lock = asyncio.Lock()
        setattr(app_state, _APP_STATE_WRITE_LOCK_ATTR, new_lock)
        setattr(app_state, _APP_STATE_WRITE_LOOP_ATTR, running_loop)
        return new_lock
    return existing_lock


def _get_idempotency_state(app_state: Any) -> _IdempotencyState:
    """Lazily fetch-or-create the per-app idempotency state.

    Same event-loop-binding reasoning as `_get_write_lock`, but with an
    important wrinkle: the LRU `cache` MUST survive loop changes, else
    `TestClient` (which creates a fresh `asyncio` loop per request) and
    any other multi-loop host would never see the replay it just cached.
    We therefore preserve the `_IdempotencyState` object itself and only
    rebind its `asyncio.Lock` + clear `in_flight` when the running loop
    changes. Any waiters on `in_flight` events from a dead loop went
    away with it, so dropping them here is safe.
    """
    running_loop = asyncio.get_running_loop()
    existing: _IdempotencyState | None = getattr(app_state, _APP_STATE_IDEMPOTENCY_ATTR, None)
    existing_loop = getattr(app_state, _APP_STATE_IDEMPOTENCY_LOOP_ATTR, None)
    if existing is None:
        new_state = _IdempotencyState()
        setattr(app_state, _APP_STATE_IDEMPOTENCY_ATTR, new_state)
        setattr(app_state, _APP_STATE_IDEMPOTENCY_LOOP_ATTR, running_loop)
        return new_state
    if existing_loop is not running_loop:
        existing.rebind_to_current_loop()
        setattr(app_state, _APP_STATE_IDEMPOTENCY_LOOP_ATTR, running_loop)
    return existing


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
    """Validate model_id to prevent path traversal and gateway route ambiguity."""
    if ".." in model_id or "\\" in model_id or not _MODEL_ID_PATTERN.match(model_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_model_id", "message": "Model ID contains invalid characters."},
        )
    # The gateway's `GET /v1/configs/models/{*id}` uses a `/status` suffix
    # as an in-band signal for the worker-ack status endpoint. A model id
    # ending in `/status` would collide with that dispatcher and make the
    # config-read URL ambiguous. Reject at ingest rather than paper over
    # it on the gateway side.
    if model_id == "status" or model_id.endswith("/status"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model_id",
                "message": "Model ID must not end with '/status' (reserved for the gateway status endpoint).",
            },
        )


def _load_filesystem_yaml(models_dir: Any, model_name: str) -> str | None:
    """Load raw YAML for a filesystem-backed model (blocking; call via to_thread).

    Tries the canonical `{model_name with / -> __}.yaml` filename first,
    then falls back to scanning every `*.yaml` in the directory and
    matching on the `sie_id` / `name` field inside each file.
    """
    model_path = models_dir / f"{model_name.replace('/', '__')}.yaml"
    if model_path.exists():
        return model_path.read_text()

    for p in models_dir.glob("*.yaml"):
        try:
            content = p.read_text()
            data = yaml.safe_load(content)
        except yaml.YAMLError:
            continue
        if data and (data.get("sie_id") == model_name or data.get("name") == model_name):
            return content
    return None


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


def _require_model_registry(request: Request) -> ModelRegistry:
    """Return the app's ``ModelRegistry`` or raise ``503`` if it's not loaded.

    The app keeps serving when ``ModelRegistry`` initialization fails (see
    ``app_factory._model_registry``) so ``/readyz`` can correctly report
    ``503``. Config routes used to dereference ``app.state.model_registry``
    unconditionally and return ``500 AttributeError`` in that same state,
    which produced inconsistent probe-vs-route behaviour. Using this helper
    keeps every handler on the same "registry absent → 503" contract and
    makes the error structured rather than an unhandled exception.
    """
    registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "registry_unavailable",
                "message": (
                    "ModelRegistry failed to initialize (e.g. bad bundle/model YAML). "
                    "See /readyz and config service logs; fix the on-disk state and restart."
                ),
            },
        )
    return registry


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List all registered model IDs with their profiles and source."""
    _check_read_auth(request)
    model_registry = _require_model_registry(request)
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
    model_registry = _require_model_registry(request)

    if not model_registry.model_exists(model_id):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "model_id": model_id,
                "message": f"Model '{model_id}' does not exist in the catalog.",
            },
        )

    content = (await asyncio.to_thread(config_store.read_model, model_id)) if config_store else None
    if content:
        return Response(content=content, media_type="application/x-yaml")

    raw_yaml = await asyncio.to_thread(_load_filesystem_yaml, model_registry.models_dir, model_id)
    if raw_yaml is not None:
        return Response(content=raw_yaml, media_type="application/x-yaml")

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

    model_registry = _require_model_registry(request)
    nats_publisher: NatsPublisher | None = getattr(request.app.state, "nats_publisher", None)
    config_store: ConfigStore | None = getattr(request.app.state, "config_store", None)

    # Check NATS availability (required for config distribution)
    if nats_publisher and not nats_publisher.connected:
        raise HTTPException(
            status_code=503,
            detail={"error": "nats_unavailable", "message": "NATS not connected -- cannot distribute config changes."},
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

    # Idempotency-Key support (checked after body read for payload hash)
    idempotency_key = request.headers.get("Idempotency-Key")
    body_hash = hashlib.sha256(body).hexdigest()
    idem = _get_idempotency_state(request.app.state) if idempotency_key else None
    if idempotency_key and idem is not None:
        # Track whether we already waited on an in-flight event. If we
        # did, and the subsequent cache lookup misses (the original
        # request's entry was LRU-evicted in the meantime), we must NOT
        # re-execute the write — that would double-apply the config and
        # bump the epoch twice. Return a synthesized 200 "already
        # processed" response instead; the idempotency contract says
        # "never execute twice for the same key", it says nothing about
        # guaranteeing the original body is still replayable.
        already_waited = False
        while True:
            async with idem.lock:
                cached = idem.cache.get(idempotency_key)
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
                in_flight_event = idem.in_flight.get(idempotency_key)
                if in_flight_event is None:
                    if already_waited:
                        # Woken by a completer but the response was
                        # evicted before we grabbed the lock. Replay-safe
                        # synthesized response: tell the caller the write
                        # was already applied and they should trust the
                        # prior success. Not 200-from-cache because we
                        # can't echo created_profiles faithfully — the
                        # caller should re-read GET /v1/configs/models/{id}
                        # if they need the post-state.
                        return Response(
                            content=orjson.dumps(
                                {
                                    "error": "idempotent_replay_evicted",
                                    "idempotency_key": idempotency_key,
                                    "message": (
                                        "A prior request with this Idempotency-Key completed successfully, "
                                        "but its cached response was evicted. The write was applied exactly once; "
                                        "query GET /v1/configs/models/{id} to see the current state."
                                    ),
                                }
                            ).decode(),
                            status_code=200,
                            media_type="application/json",
                        )
                    # Genuinely first request for this key.
                    idem.in_flight[idempotency_key] = asyncio.Event()
                    break
            already_waited = True
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

        # Validate model ID before touching the registry
        model_id = config.get("sie_id", "")
        if model_id:
            _validate_model_id(model_id)

        # -------------------------------------------------------------
        # Serialize the full write critical section. See `_get_write_lock`
        # docstring for the ordering constraints (persist → mutate
        # registry → increment_epoch → publish). Validation runs inside
        # the lock too, so a concurrent write cannot sneak a conflicting
        # profile in between validation and mutation.
        # -------------------------------------------------------------
        write_lock = _get_write_lock(request.app.state)
        async with write_lock:
            # 1. Validate without mutating the registry so a 422 (invalid
            #    YAML shape, unroutable adapter, append-only conflict)
            #    never leaves the registry inconsistent with disk.
            try:
                created_profiles, skipped_profiles, affected_bundles = model_registry.validate_model_config(config)
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "validation_error", "details": [{"message": str(e)}]},
                ) from e

            # 2. 409 conflict detection against what's already on disk.
            #    Only runs for pure-replay writes (no new profiles would be
            #    created). The `skipped_profiles` list already filters out
            #    identical-content replays via `validate_model_config`, so
            #    if we get here with `skipped_profiles` and no `created`,
            #    the incoming profile matches the registry byte-for-byte
            #    (canonical-hash-fields-wise). The disk check picks up the
            #    rare case where the registry is ahead of disk after a
            #    previous crashed write, and we still want 409 rather than
            #    silently accepting a divergent body.
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
                        pass

            # 3. Persist FIRST, then mutate the in-memory registry.
            #    A disk-write failure must abort the write before the
            #    in-memory state changes. The registry rebuilds from disk
            #    on restart; if we mutated the registry first and then
            #    failed to persist, a subsequent restart would silently
            #    drop the model while workers had already observed it.
            #
            #    Merge invariants (append-only):
            #      - Existing top-level fields that the new body omits
            #        are PRESERVED (minimal profile-append bodies must
            #        not erase `description`, `default_bundle`, etc.).
            #      - A new body may introduce top-level fields that the
            #        stored document doesn't have.
            #      - If both sides set the same non-`profiles` top-level
            #        field with different values, we reject with 409 —
            #        mutating existing metadata is NOT append-only, so
            #        it must go through an explicit operation, not get
            #        smuggled into a profile-append request.
            epoch = 0
            config_yaml = ""
            if config_store and created_profiles:
                existing_yaml = await asyncio.to_thread(config_store.read_model, model_id)
                if existing_yaml:
                    try:
                        existing_config = yaml.safe_load(existing_yaml) or {}
                    except yaml.YAMLError:
                        existing_config = {}
                    if not isinstance(existing_config, dict):
                        existing_config = {}

                    merged_config: dict[str, Any] = dict(existing_config)

                    conflicting_fields: list[str] = []
                    for key, new_value in config.items():
                        if key == "profiles":
                            continue
                        if key in merged_config and merged_config[key] != new_value:
                            conflicting_fields.append(key)
                        else:
                            merged_config[key] = new_value

                    if conflicting_fields:
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error": "content_conflict",
                                "model_id": model_id,
                                "conflicting_fields": sorted(conflicting_fields),
                                "message": (
                                    f"Top-level field(s) {sorted(conflicting_fields)} already "
                                    "exist with different values. Config API is append-only; "
                                    "mutating existing metadata is not supported via this endpoint."
                                ),
                            },
                        )

                    merged_profiles = dict(existing_config.get("profiles", {}))
                    merged_profiles.update(config.get("profiles", {}))
                    merged_config["profiles"] = merged_profiles
                else:
                    merged_config = dict(config)

                config_yaml = yaml.dump(merged_config, default_flow_style=False, sort_keys=False)

                # Disk write BEFORE registry mutation.
                await asyncio.to_thread(config_store.write_model, model_id, config_yaml)

            # 4. Commit to in-memory registry now that disk is durable.
            #    Validation already ran above; this call is expected to
            #    succeed. If it does raise (e.g. OOM, bug), the on-disk
            #    YAML is still the source of truth and will be picked up
            #    by the next `reload()` or service restart.
            try:
                applied_created, applied_skipped, applied_affected = model_registry.add_model_config(config)
            except ValueError as e:  # pragma: no cover -- defense in depth
                raise HTTPException(
                    status_code=422,
                    detail={"error": "validation_error", "details": [{"message": str(e)}]},
                ) from e
            # The triple from the real mutation is authoritative in case
            # the registry rebuilt bundle mappings differently.
            created_profiles = applied_created
            skipped_profiles = applied_skipped
            affected_bundles = applied_affected

            # No-ConfigStore path: after the registry merge we have the
            # authoritative full model dict in memory. Serialize *that*
            # (not the raw incremental request body) so the NATS delta
            # carries the complete merged YAML. A fresh gateway that
            # receives this delta — or that bootstraps via
            # `/v1/configs/export` below — gets the full model instead of
            # just the newly-appended profiles.
            if not config_store and created_profiles:
                merged = model_registry.get_full_config(model_id)
                if merged is not None:
                    config_yaml = yaml.dump(merged, default_flow_style=False, sort_keys=False)
                else:
                    # Registry did not retain the model (shouldn't happen,
                    # defensive). Fall back to the raw request body so
                    # we don't publish an empty payload.
                    config_yaml = body.decode()

            # 5. Increment epoch atomically. This is a read-modify-write
            #    on `ConfigStore`, so it MUST stay inside `write_lock`.
            if config_store and created_profiles:
                epoch = await asyncio.to_thread(config_store.increment_epoch)

            # Refresh the `sie_config_models_total` gauge now that the
            # registry has been mutated. Cheap (in-memory dict scan)
            # and inside the write lock so the count we report is the
            # one that corresponds to the epoch we just bumped.
            if created_profiles:
                total = len(model_registry.list_models())
                api_count = len(await asyncio.to_thread(config_store.list_models)) if config_store else 0
                sie_metrics.update_models_gauge(
                    api_count=api_count,
                    filesystem_count=max(total - api_count, 0),
                )

            # 6. Compute bundle config hashes and publish NATS. Publish
            #    stays inside the lock so the `(bundle_id, epoch)` pairs
            #    reach the wire in strict monotonic order per bundle —
            #    gateways rely on that for their `ConfigEpoch` invariant.
            bundle_config_hashes = {}
            for bundle_id in affected_bundles:
                bundle_config_hashes[bundle_id] = model_registry.compute_bundle_config_hash(bundle_id)

            nats_publish_failed = False
            partial_publish_failed_bundles: list[str] = []
            if nats_publisher and nats_publisher.connected and created_profiles:
                try:
                    await nats_publisher.publish_config_notification(
                        model_id=model_id,
                        profiles_added=created_profiles,
                        affected_bundles=affected_bundles,
                        bundle_config_hashes=bundle_config_hashes,
                        epoch=epoch,
                        model_config_yaml=config_yaml,
                    )
                except PartialPublishError as e:
                    logger.exception(
                        "Partial NATS publish for model %s: %s/%s bundles failed: %s",
                        model_id,
                        len(e.failed_bundles),
                        e.total,
                        e.failed_bundles,
                    )
                    nats_publish_failed = True
                    partial_publish_failed_bundles = e.failed_bundles
                except Exception:
                    logger.exception("Failed to publish NATS notification for model %s", model_id)
                    nats_publish_failed = True

        router_id = nats_publisher.router_id if nats_publisher else "standalone"
        warnings: list[str] = []
        if partial_publish_failed_bundles:
            warnings.append(
                "nats_publish_partial: Config persisted but NATS notification failed for bundle(s) "
                f"{partial_publish_failed_bundles}. Workers on those bundles will stay on the previous "
                "epoch until the gateway's periodic epoch poll triggers a re-export."
            )
        elif nats_publish_failed:
            warnings.append(
                "nats_publish_failed: Config persisted but NATS notification failed. Workers may not be updated."
            )

        routable_bundles_by_profile = {}
        for profile_name in created_profiles:
            routable_bundles_by_profile[profile_name] = affected_bundles

        status_code = 201 if created_profiles else 200
        response_body: dict[str, Any] = {
            "model_id": model_id,
            "created_profiles": created_profiles,
            "existing_profiles_skipped": skipped_profiles,
            "warnings": warnings,
            "routable_bundles_by_profile": routable_bundles_by_profile,
            "router_id": router_id,
        }

        response_json = orjson.dumps(response_body).decode()

        if idempotency_key and idem is not None:
            async with idem.lock:
                idem.cache[idempotency_key] = (status_code, response_json, body_hash)
                idem.cache.move_to_end(idempotency_key)
                while len(idem.cache) > _MAX_IDEMPOTENCY_CACHE_SIZE:
                    idem.cache.popitem(last=False)
                event = idem.in_flight.pop(idempotency_key, None)
                if event is not None:
                    event.set()

    except Exception:
        # Clean up in-flight marker on any error so waiters aren't stuck forever
        if idempotency_key and idem is not None:
            async with idem.lock:
                event = idem.in_flight.pop(idempotency_key, None)
                if event is not None:
                    event.set()
        raise

    # Emit audit log
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


@router.post("/resolve")
async def resolve_config(request: Request) -> Response:
    """Resolve a model spec to its bundle and routing info.

    Accepts a JSON body with 'model' (required) and optional 'bundle' override.
    Returns the resolved bundle, compatible bundles, and profile names.
    """
    _check_read_auth(request)
    model_registry = _require_model_registry(request)

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
    """List all known bundles."""
    _check_read_auth(request)
    model_registry = _require_model_registry(request)

    bundles = []
    for bundle_name in model_registry.list_bundles():
        info = model_registry.get_bundle_info(bundle_name)
        if not info:
            continue

        bundles.append(
            {
                "bundle_id": info.name,
                "priority": info.priority,
                "adapter_count": len(info.adapters),
                "source": "filesystem",
            }
        )

    return {"bundles": bundles}


@router.get("/bundles/{bundle_id}")
async def get_bundle(request: Request, bundle_id: str) -> Response:
    """Return bundle metadata YAML."""
    _check_read_auth(request)
    model_registry = _require_model_registry(request)

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


@router.get("/epoch")
async def latest_epoch(request: Request) -> dict[str, int | str]:
    """Return the current config epoch counter and bundle-set fingerprint.

    Lightweight complement to `GET /v1/configs/export`. Called periodically
    by the Rust gateway's `state::config_poller` to detect drift on TWO
    independent axes:

    - ``epoch`` (int, monotonic): bumped on every model-config write. If the
      remote value is ahead of the gateway's local value, the gateway
      triggers a fresh ``/v1/configs/export`` fetch to catch up. Closes the
      NATS Core pub/sub delta-loss gap (pub/sub has no replay, so a
      disconnected gateway could otherwise silently fall behind).
    - ``bundles_hash`` (hex sha256, may be empty): fingerprint over the
      registry's full bundle surface (see ``ModelRegistry.compute_bundles_hash``).
      Bundles are filesystem artifacts baked into this image; their epoch is
      effectively "redeploy time", which the model epoch counter does not
      observe. The gateway re-fetches ``/v1/configs/bundles`` whenever this
      hash changes, independent of the model epoch — without it, a sie-config
      redeploy that adds a bundle would not propagate to the gateway until a
      coincidental model write or a manual gateway restart.

    Intentionally uses read auth rather than admin auth — the epoch is a
    monotonic integer counter and the hash is a one-way digest; neither
    reveals anything sensitive and broadening the caller set (SRE tooling,
    monitoring) beyond the admin-token holder keeps operational surface
    cheap.
    """
    _check_read_auth(request)
    config_store: ConfigStore | None = getattr(request.app.state, "config_store", None)
    epoch = (await asyncio.to_thread(config_store.read_epoch)) if config_store else 0
    # Stay tolerant of a missing registry — `/epoch` is the gateway's
    # liveness signal for sie-config and must keep answering even during a
    # registry init failure (see test_epoch_works_without_registry). Empty
    # `bundles_hash` is the documented "nothing to sync" sentinel; the
    # gateway will not trigger a bundle re-fetch in that state.
    model_registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)
    bundles_hash = model_registry.compute_bundles_hash() if model_registry is not None else ""
    return {"epoch": epoch, "bundles_hash": bundles_hash}


@router.get("/export")
async def export_snapshot(request: Request) -> Response:
    """Export full config snapshot for gateway bootstrap.

    Returns a JSON snapshot with all model configs, their raw YAML,
    and the current epoch. Used by the Rust gateway for bootstrap.

    Consistency: this endpoint MUST return a `(epoch, models)` pair that
    corresponds to a real serialization point of the writer. The danger
    case we avoid is `epoch_returned > epoch_actually_reflected_by_models`
    (i.e. "this model was added in epoch E, but the snapshot says we're
    at E+1 and the model is missing from the `models` list"). If that
    escapes to a gateway, the poller will see `remote == local`, log
    "in sync", and silently leave the gateway missing the model forever
    until the next restart or NATS delta.

    We enforce consistency the cheapest way: take the per-app write
    lock around the snapshot. Writers serialize on the same lock so
    the snapshot is always "between writes". Export is a rare
    operation (gateway bootstrap + drift recovery), so the brief
    write-path blockage is acceptable. Read-path handlers like
    `/epoch`, `/models`, `/bundles` remain fully concurrent.
    """
    _check_write_auth(request)  # admin-only
    model_registry = _require_model_registry(request)
    config_store: ConfigStore | None = getattr(request.app.state, "config_store", None)

    write_lock = _get_write_lock(request.app.state)
    async with write_lock:
        epoch = (await asyncio.to_thread(config_store.read_epoch)) if config_store else 0

        models = []
        for model_name in model_registry.list_models():
            model_info = model_registry.get_model_info(model_name)
            raw_yaml = (await asyncio.to_thread(config_store.read_model, model_name)) if config_store else None

            if raw_yaml is None:
                raw_yaml = await asyncio.to_thread(_load_filesystem_yaml, model_registry.models_dir, model_name)

            if raw_yaml:
                try:
                    model_config = yaml.safe_load(raw_yaml) or {"sie_id": model_name}
                except yaml.YAMLError:
                    model_config = None
            else:
                model_config = None

            # Fall back to the registry's in-memory merged config whenever
            # we don't have authoritative raw YAML on disk. This covers
            # no-ConfigStore deployments (default) and API-added models,
            # both of which previously exported as `{"sie_id": ...}` and
            # caused a fresh gateway's bootstrap to drop all profiles.
            if not model_config or not model_config.get("profiles"):
                full = model_registry.get_full_config(model_name)
                if full:
                    model_config = full
                    if not raw_yaml:
                        raw_yaml = yaml.dump(full, default_flow_style=False, sort_keys=False)
                elif not model_config:
                    model_config = {"sie_id": model_name}

            models.append(
                {
                    "model_id": model_name,
                    "model_config": model_config,
                    "raw_yaml": raw_yaml,
                    "affected_bundles": model_info.bundles if model_info else [],
                }
            )

    snapshot = {
        "snapshot_version": 1,
        "epoch": epoch,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": models,
    }
    return Response(content=orjson.dumps(snapshot), media_type="application/json")
