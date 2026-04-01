from fastapi import HTTPException, status

from sie_server.observability.gpu import get_worker_gpu_type
from sie_server.types.responses import ErrorCode


def validate_machine_profile_header(x_machine_profile: str | None) -> None:
    """Validate X-SIE-MACHINE-PROFILE header against worker identity.

    When a request specifies a machine profile via X-SIE-MACHINE-PROFILE header,
    the worker validates that it matches its identity. The worker's identity is:
    - In K8s: SIE_MACHINE_PROFILE env var (e.g., "l4-spot")
    - Standalone: Detected GPU type (e.g., "l4")

    This catches routing errors early and provides useful error messages.

    Args:
        x_machine_profile: Machine profile from request header.

    Raises:
        HTTPException: 400 if profile doesn't match, with actual vs requested info.
    """
    import os

    if not x_machine_profile:
        return  # No profile specified, allow request

    # Worker identity: env var if set, else detected GPU type (for standalone workers)
    detected_gpu = get_worker_gpu_type()
    worker_identity = os.environ.get("SIE_MACHINE_PROFILE") or detected_gpu

    # If no identity (no env var and no GPU), reject profile-targeted requests
    if worker_identity is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": f"Request targets machine profile '{x_machine_profile}' but this worker has no GPU "
                "and SIE_MACHINE_PROFILE is not set. "
                "Route requests to a GPU worker or remove the X-SIE-MACHINE-PROFILE header.",
                "requested_profile": x_machine_profile,
                "worker_identity": None,
            },
        )

    # Normalize for comparison
    requested = x_machine_profile.lower().strip()
    actual = worker_identity.lower().strip()

    if requested != actual:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": f"Request targets machine profile '{x_machine_profile}' but this worker is '{worker_identity}'. "
                "This may indicate a routing error. Check your router configuration.",
                "requested_profile": x_machine_profile,
                "worker_identity": worker_identity,
            },
        )
