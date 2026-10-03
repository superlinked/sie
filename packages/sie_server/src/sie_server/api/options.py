from typing import Any

from fastapi import HTTPException, status

from sie_server.config.model import ModelConfig, ResolvedProfile
from sie_server.core.runtime_options import InvalidOverflowPolicyError, merge_runtime_options_with_profile
from sie_server.types.responses import ErrorCode


def resolve_runtime_options(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
    span: Any,
) -> dict[str, Any]:
    """Resolve runtime options from profile + request overrides.

    Resolves the profile's runtime dict via config.resolve_profile(), then
    merges per-request overrides on top. The "profile" key is consumed here
    and not passed through.

    Args:
        config: Model configuration with profiles.
        request_options: Raw options dict from the request (may contain "profile" key).
        span: OpenTelemetry span for error attribution.

    Returns:
        Merged options dict ready to pass through to the worker/adapter.

    Raises:
        HTTPException: 400 if profile name is invalid.
    """
    merged, _ = resolve_runtime_options_with_profile(config, request_options, span)
    return merged


def resolve_runtime_options_with_profile(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
    span: Any,
) -> tuple[dict[str, Any], ResolvedProfile]:
    """Resolve runtime options and retain the selected profile for encode validation."""
    try:
        merged, resolved_profile = merge_runtime_options_with_profile(config, request_options)
    except ValueError as e:
        # The same check the queue worker applies (core.runtime_options).
        invalid = "invalid_overflow_policy" if isinstance(e, InvalidOverflowPolicyError) else "invalid_profile"
        span.set_attribute("error", invalid)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": str(e),
            },
        ) from e

    return merged, resolved_profile
