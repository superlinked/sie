from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

router = APIRouter(prefix="/v1", tags=["models"])

# Wire-level state strings exposed in ``ModelInfo.state``. Mirrors the
# server-side ``ModelRegistry`` state machine plus the terminal ``failed``
# branch added for non-retryable load failures.
ModelStateStr = Literal["available", "loading", "loaded", "unloading", "failed"]


class ProfileInfo(BaseModel):
    """Information about a profile."""

    is_default: bool = False


class ModelLoadError(BaseModel):
    """Diagnostic detail for a recorded load failure.

    Surfaced in :class:`ModelInfo` when the registry has a sticky
    failure for the model. Attributes mirror the server-side
    :class:`sie_server.core.load_errors.LoadFailure`.
    """

    code: str
    """Stable enum value (``GATED``, ``OOM``, ...) for client routing."""

    message: str
    """Human-readable error summary, including the underlying exception."""

    attempts: int
    """How many load attempts have failed so far."""

    permanent: bool
    """True when the failure will not auto-retry; operator must intervene."""


class ModelInfo(BaseModel):
    """Information about a model."""

    name: str
    inputs: list[str]
    outputs: list[str]
    dims: dict[str, int]
    loaded: bool
    """Backwards-compatible boolean. Prefer ``state`` for full lifecycle."""

    state: ModelStateStr = "available"
    """Lifecycle state including the terminal ``failed`` branch."""

    last_error: ModelLoadError | None = None
    """Recorded load failure (when ``state == 'failed'``), else ``None``."""

    max_sequence_length: int | None = None
    profiles: dict[str, ProfileInfo] = {}


def _resolve_state_and_error(
    registry: "ModelRegistry | Any",
    name: str,
) -> tuple[ModelStateStr, ModelLoadError | None]:
    """Compute ``(state, last_error)`` for a model from the registry.

    Mirrors the ws.py state precedence (loading > unloading > loaded >
    failed > available) and produces the diagnostic payload for
    ``last_error`` when a sticky failure exists. The ``Any`` union in
    the signature accommodates the ``MagicMock(spec=ModelRegistry)``
    fixtures used in API tests.
    """
    is_loading = registry.is_loading(name)
    is_unloading = registry.is_unloading(name)
    is_loaded = registry.is_loaded(name)
    is_failed = registry.is_failed(name)
    failure = registry.get_failure(name)

    state: ModelStateStr
    if is_loading:
        state = "loading"
    elif is_unloading:
        state = "unloading"
    elif is_loaded:
        state = "loaded"
    elif is_failed:
        state = "failed"
    else:
        state = "available"

    last_error: ModelLoadError | None = None
    if failure is not None:
        last_error = ModelLoadError(
            code=failure.error_class.value,
            message=failure.message,
            attempts=failure.attempts,
            permanent=failure.is_permanent,
        )
    return state, last_error


class ModelsListResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelInfo]


@router.get("/models")
async def list_models(http_request: Request) -> ModelsListResponse:
    """List all available models.

    Args:
        http_request: FastAPI request object (for accessing app state).

    Returns:
        List of all models with their info.
    """
    registry = http_request.app.state.registry

    models = []
    for name in registry.model_names:
        config = registry.get_config(name)
        profiles = {
            pname: ProfileInfo(
                is_default=(pname == "default"),
            )
            for pname in config.profiles
        }
        state, last_error = _resolve_state_and_error(registry, name)
        models.append(
            ModelInfo(
                name=config.sie_id,
                inputs=config.inputs.to_list(),
                outputs=config.outputs,
                dims=config.dims,
                loaded=registry.is_loaded(name),
                state=state,
                last_error=last_error,
                max_sequence_length=config.max_sequence_length,
                profiles=profiles,
            )
        )

    return ModelsListResponse(models=models)


@router.get(
    "/models/{model:path}",
    responses={404: {"description": "Model not found"}},
)
async def get_model(model: str, http_request: Request) -> ModelInfo:
    """Get details for a specific model.

    Args:
        model: Model name.
        http_request: FastAPI request object (for accessing app state).

    Returns:
        Model info.

    Raises:
        HTTPException: 404 if model not found.
    """
    registry = http_request.app.state.registry

    if not registry.has_model(model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.MODEL_NOT_FOUND.value,
                "message": f"Model '{model}' not found",
            },
        )

    config = registry.get_config(model)
    profiles = {
        pname: ProfileInfo(
            is_default=(pname == "default"),
        )
        for pname in config.profiles
    }
    state, last_error = _resolve_state_and_error(registry, model)
    return ModelInfo(
        name=config.sie_id,
        inputs=config.inputs.to_list(),
        outputs=config.outputs,
        dims=config.dims,
        loaded=registry.is_loaded(model),
        state=state,
        last_error=last_error,
        max_sequence_length=config.max_sequence_length,
        profiles=profiles,
    )
