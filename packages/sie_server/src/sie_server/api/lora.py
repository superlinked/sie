"""Declared LoRA selection and request lifetime on direct generation routes."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from sie_server.adapters._generation_base import aclose_with_error_precedence

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig
    from sie_server.core.registry import ModelRegistry


def validate_generation_lora(config: ModelConfig, value: Any, *, profile: str = "default") -> str | None:
    """Validate only names declared by the selected profile, before any load."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or value not in config.resolve_profile(profile).lora_names():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": "The selected profile does not declare this LoRA adapter",
                "param": "lora_adapter",
            },
        )
    return value


async def ensure_generation_lora(
    registry: ModelRegistry, model: str, name: str | None, *, n: int | None = None
) -> None:
    """Admit a declared adapter or return the same cold-load retry contract as encoders."""
    if name is None:
        return
    adapter = registry.get(model)
    runtime = adapter.lora_runtime_config()
    if runtime is not None and set(runtime.aliases.values()) - runtime.preloaded and n is not None and n > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": "Deferred LoRA generation supports only n=1",
                "param": "n",
            },
        )
    try:
        ready, _ = await registry.ensure_lora_loaded_async(model, name)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": "The selected profile does not support this LoRA adapter",
                "param": "lora_adapter",
            },
        ) from exc
    except KeyError:
        ready = False
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"code": "LORA_LOADING", "message": "The selected LoRA is loading; retry the request"},
            headers={"Retry-After": str(registry.lora_retry_after_s(model, name))},
        )


class LeasedStreamingResponse(StreamingResponse):
    """Release a pre-acquired lease even when ASGI streaming fails before its first yield."""

    def __init__(self, response: StreamingResponse, lifetime: AsyncExitStack) -> None:
        super().__init__(
            response.body_iterator,
            status_code=response.status_code,
            media_type=response.media_type,
            background=response.background,
        )
        self.raw_headers = response.raw_headers
        self._response = response
        self._lifetime = lifetime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._response(scope, receive, send)
        finally:
            try:
                await aclose_with_error_precedence(
                    self._response.body_iterator,
                    outcome_selected=True,
                    context="generation child SSE iterator",
                )
            finally:
                await self._lifetime.aclose()
