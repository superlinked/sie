from typing import Any

from fastapi import HTTPException, status

from sie_server.config.model import ModelConfig
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
    profile_name = request_options.get("profile") if request_options else None

    try:
        resolved = config.resolve_profile(profile_name or "default")
    except ValueError as e:
        span.set_attribute("error", "invalid_profile")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": str(e),
            },
        ) from e

    # Start with profile runtime options
    merged = dict(resolved.runtime)

    # Merge request-level overrides (excluding the "profile" key itself)
    if request_options:
        merged |= {k: v for k, v in request_options.items() if k != "profile"}
    return merged
