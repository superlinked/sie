"""Which registry entry serves a request on this server, and the headers that say so.

A request for the bare name of a model with the ``fallback`` routing policy is
served by the local profile while it can serve. When the local path refuses
before accepting the work, for one of the policy's triggers, the model's remote
fallback profile serves instead and the local load starts. A request that
names a profile is served as written, and ``X-SIE-Remote: forbid`` keeps a
request off every remote profile.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Header, HTTPException, Request, status

from sie_server.api.helpers import SERVED_BY_HEADER, UPSTREAM_HEADER, ModelStateChecker, queue_full_exception
from sie_server.config.hybrid_admission import hybrid_request_refusal
from sie_server.config.routing import validate_model_routing
from sie_server.config.upstreams import remote_serving_enabled
from sie_server.core.loader import serves_remotely
from sie_server.core.profile_identity import runtime_instance_id
from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from sie_server.config.model import FallbackTrigger, ModelConfig, RoutingConfig
    from sie_server.core.registry import ModelRegistry

REMOTE_HEADER = "X-SIE-Remote"
FALLBACK_REASON_HEADER = "X-SIE-Fallback-Reason"
FALLBACK_ERROR_HEADER = "X-SIE-Fallback-Error"

_FORBID = "forbid"
_REMOTE_LOAD_RETRY_AFTER_S = 1
_ERROR_CODES = frozenset(code.value for code in ErrorCode)
_OPENAI_ERROR_CODES = {
    "server_overloaded": ErrorCode.QUEUE_FULL.value,
    "invalid_request": ErrorCode.INVALID_INPUT.value,
}
_TRIGGER_BY_REFUSAL: dict[str, FallbackTrigger] = {
    ErrorCode.MODEL_LOADING.value: "model_loading",
    ErrorCode.MODEL_NOT_LOADED.value: "model_loading",
    ErrorCode.QUEUE_FULL.value: "saturated",
    ErrorCode.MODEL_LOAD_FAILED.value: "unhealthy",
}


@dataclass(frozen=True)
class ServingRoute:
    """The registry entry that serves a request, and the upstream it calls, if any."""

    key: str
    upstream: str | None = None
    fallback_reason: FallbackTrigger | None = None
    local_refusal: HTTPException | None = None
    """What the local path answered before a bridged request went to the remote profile."""

    def headers(self) -> dict[str, str]:
        """Disclosure headers for a response served through this route."""
        if self.upstream is None:
            return {SERVED_BY_HEADER: "local"}
        headers = {SERVED_BY_HEADER: "remote", UPSTREAM_HEADER: self.upstream}
        if self.fallback_reason is not None:
            headers[FALLBACK_REASON_HEADER] = self.fallback_reason
        return headers

    def refusal_after(self, status_code: int, code: object) -> HTTPException | None:
        """The local refusal that answers a bridged request whose remote attempt failed, or ``None``.

        Every failure after the bridge decision is the remote side's, because
        the request has passed this server's own validation by then. The
        replacement keeps the local refusal's status, body and ``Retry-After``;
        its headers name both outcomes.
        """
        local = self.local_refusal
        if local is None or self.fallback_reason is None:
            return None
        headers = {
            **(local.headers or {}),
            SERVED_BY_HEADER: "local",
            FALLBACK_REASON_HEADER: self.fallback_reason,
            FALLBACK_ERROR_HEADER: _fallback_error(status_code, code),
        }
        return HTTPException(status_code=local.status_code, detail=local.detail, headers=headers)


def _fallback_error(status_code: int, code: object) -> str:
    """The server error code a failed remote attempt answered with."""
    if isinstance(code, str):
        if code.upper() in _ERROR_CODES:
            return code.upper()
        if code in _OPENAI_ERROR_CODES:
            return _OPENAI_ERROR_CODES[code]
    if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return ErrorCode.INFERENCE_ERROR.value
    return ErrorCode.INVALID_INPUT.value


def error_code(error: HTTPException) -> str | None:
    """The error code of a native ``{code, ...}`` or OpenAI ``{"error": {code, ...}}`` detail."""
    detail = error.detail
    if not isinstance(detail, dict):
        return None
    nested = detail.get("error")
    code = (nested if isinstance(nested, dict) else detail).get("code")
    return code if isinstance(code, str) else None


def fallback_refusal(request: Request, status_code: int, code: object) -> HTTPException | None:
    """The local refusal that replaces this request's failed bridged attempt, or ``None``."""
    route = getattr(request.state, "serving_route", None)
    return route.refusal_after(status_code, code) if isinstance(route, ServingRoute) else None


async def remote_routing(
    request: Request,
    x_sie_remote: Annotated[
        str | None,
        Header(
            alias=REMOTE_HEADER,
            description=(
                "The only accepted value is `forbid`. It keeps the request off remote upstreams: local capacity "
                "serves or refuses it, and a model served only by an upstream answers 400. Any other value answers 400."
            ),
        ),
    ] = None,
) -> AsyncIterator[None]:
    """Router dependency of every route that serves a model.

    It declares ``X-SIE-Remote``, which :func:`route_request` reads, and
    answers a bridged request whose remote attempt raised with the local
    refusal that attempt replaced.
    """
    _ = x_sie_remote
    try:
        yield
    except HTTPException as error:
        error.headers = {**(error.headers or {}), "X-SIE-Runtime-Instance": runtime_instance_id()}
        refusal = fallback_refusal(request, error.status_code, error_code(error))
        if refusal is None:
            raise
        raise refusal from error


async def route_request(
    request: Request,
    model: str,
    span: Span,
    *,
    serving_key: str | None = None,
    profile: object = None,
    queued_items: int | None = None,
    request_options: Mapping[str, object] | None = None,
) -> ServingRoute:
    """Choose the registry entry that serves a request for ``model``, and make it ready to serve.

    Args:
        request: The HTTP request, for ``X-SIE-Remote`` and the registry.
        model: Registry key the request names.
        span: Span for error attributes.
        serving_key: The entry that serves ``model`` locally when it is not
            ``model`` itself, such as a grammar-safe profile.
        profile: The profile the request explicitly names in its options.
            Every named profile, including ``default``, is served as written.
        queued_items: Items the request adds to the model's queue. With it,
            the ``saturated`` trigger sees a full queue before submission.

    Raises:
        HTTPException: The answer the caller receives when no entry serves now.
    """
    registry: ModelRegistry = request.app.state.registry
    forbid = _remote_forbidden(request)
    ModelStateChecker(registry, model, span).check_exists()
    routing = None if isinstance(profile, str) else registry.get_config(model).routing
    key = serving_key or model
    config = registry.get_config(key)
    if serves_remotely(config):
        if forbid:
            span.set_attribute("error", "remote_forbidden")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": f"Model '{key}' is served only by an upstream, which {REMOTE_HEADER}: {_FORBID} refuses",
                },
            )
        await _ready_remote(registry, key, span)
        return ServingRoute(key=key, upstream=_upstream_name(config))
    if forbid or routing is None or routing.policy != "fallback" or not remote_serving_enabled():
        await _ready_local(registry, key, span)
        return ServingRoute(key=key)
    triggers = routing.effective_triggers
    try:
        await _ready_local(registry, key, span, queued_items=queued_items if "saturated" in triggers else None)
    except HTTPException as refusal:
        trigger = _TRIGGER_BY_REFUSAL.get(error_code(refusal) or "")
        if trigger is None or trigger not in triggers:
            raise
        return await _bridge(request, registry, model, key, routing, trigger, refusal, span, request_options)
    return ServingRoute(key=key)


def _remote_forbidden(request: Request) -> bool:
    value = request.headers.get(REMOTE_HEADER)
    if value is None:
        return False
    if value == _FORBID:
        return True
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": ErrorCode.INVALID_INPUT.value, "message": f"{REMOTE_HEADER} accepts only '{_FORBID}'"},
    )


def _upstream_name(config: ModelConfig) -> str:
    return str(config.resolve_profile("default").loadtime["upstream"])


async def _ready_local(registry: ModelRegistry, key: str, span: Span, *, queued_items: int | None = None) -> None:
    checker = ModelStateChecker(registry, key, span)
    checker.check_not_unloading()
    checker.check_not_loading()
    await checker.ensure_loaded(registry.device)
    if queued_items is None:
        return
    worker = registry.get_worker(key)
    error = worker.queue_full_error(queued_items) if worker is not None else None
    if error is not None and error.limit is not None and queued_items <= error.limit:
        span.set_attribute("error", "queue_full")
        raise queue_full_exception(error)


async def _ready_remote(registry: ModelRegistry, key: str, span: Span) -> None:
    if registry.is_loaded(key) and not registry.is_unloading(key):
        return
    checker = ModelStateChecker(registry, key, span)
    checker.check_not_failed()
    if await registry.load_now(key, registry.device):
        return
    checker.check_not_failed()
    span.set_attribute("error", "model_loading")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": ErrorCode.MODEL_LOADING.value, "message": f"Model '{key}' is loading, please retry"},
        headers={"Retry-After": str(_REMOTE_LOAD_RETRY_AFTER_S)},
    )


async def _bridge(
    request: Request,
    registry: ModelRegistry,
    model: str,
    key: str,
    routing: RoutingConfig,
    trigger: FallbackTrigger,
    refusal: HTTPException,
    span: Span,
    request_options: Mapping[str, object] | None = None,
) -> ServingRoute:
    await registry.start_load_async(key, registry.device)
    remote_key = f"{model}:{routing.fallback_profile}"
    if not registry.has_model(remote_key):
        raise refusal
    route = ServingRoute(
        key=remote_key,
        upstream=_upstream_name(registry.get_config(remote_key)),
        fallback_reason=trigger,
        local_refusal=refusal,
    )
    span.set_attribute("fallback_reason", trigger)
    config = registry.get_config(model)
    if config.tasks.encode is not None or config.tasks.score is not None:
        try:
            if hybrid_request_refusal(config, request_options) is not None:
                raise ValueError("hybrid request differs from its measured contract")
            await asyncio.to_thread(
                validate_model_routing, config, device=registry.device, engine_config=registry.engine_config
            )
        except ValueError:
            replacement = route.refusal_after(status.HTTP_503_SERVICE_UNAVAILABLE, ErrorCode.INFERENCE_ERROR.value)
            if replacement is not None:
                raise replacement from None
            raise refusal from None
    try:
        await _ready_remote(registry, remote_key, span)
    except HTTPException as error:
        replacement = route.refusal_after(error.status_code, error_code(error))
        if replacement is None:
            raise
        raise replacement from error
    request.state.serving_route = route
    return route
