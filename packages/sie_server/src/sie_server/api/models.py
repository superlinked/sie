from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from sie_server.types.responses import ErrorCode

router = APIRouter(prefix="/v1", tags=["models"])


class ProfileInfo(BaseModel):
    """Information about a profile."""

    is_default: bool = False


class ModelInfo(BaseModel):
    """Information about a model."""

    name: str
    inputs: list[str]
    outputs: list[str]
    dims: dict[str, int]
    loaded: bool
    max_sequence_length: int | None = None
    profiles: dict[str, ProfileInfo] = {}


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
        models.append(
            ModelInfo(
                name=config.sie_id,
                inputs=config.inputs.to_list(),
                outputs=config.outputs,
                dims=config.dims,
                loaded=registry.is_loaded(name),
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
    return ModelInfo(
        name=config.sie_id,
        inputs=config.inputs.to_list(),
        outputs=config.outputs,
        dims=config.dims,
        loaded=registry.is_loaded(model),
        max_sequence_length=config.max_sequence_length,
        profiles=profiles,
    )
