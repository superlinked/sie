from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Environment variable names for configuration
ENV_DEVICE = "SIE_DEVICE"
ENV_MODELS_DIR = "SIE_MODELS_DIR"
ENV_MODEL_FILTER = "SIE_MODEL_FILTER"
ENV_PRELOAD_MODELS = "SIE_PRELOAD_MODELS"


@dataclass
class AppStateConfig:
    """Configuration for FastAPI app state, passed to lifespan.

    This dataclass holds all startup configuration needed by the lifespan
    context manager to initialize the model registry and related components.
    """

    models_dir: Path | str | None = None
    """Path to models directory (local path, s3://, or gs://). If None, registry starts empty."""

    device: str = "cpu"
    """Device to load models on (e.g., "cuda:0", "cpu", "mps")."""

    model_filter: list[str] | None = None
    """Optional list of model names to include. If None, all models are available."""

    preload_models: list[str] | None = None
    """Optional list of model names to eagerly load at startup. If None, all models are lazy-loaded."""

    def save_to_env_vars(self) -> None:
        """Serialize configuration to environment variables for uvicorn reload mode."""
        if self.models_dir:
            os.environ[ENV_MODELS_DIR] = str(self.models_dir)
        elif ENV_MODELS_DIR in os.environ:
            del os.environ[ENV_MODELS_DIR]

        os.environ[ENV_DEVICE] = self.device

        if self.model_filter:
            os.environ[ENV_MODEL_FILTER] = ",".join(self.model_filter)
        elif ENV_MODEL_FILTER in os.environ:
            del os.environ[ENV_MODEL_FILTER]

        if self.preload_models:
            os.environ[ENV_PRELOAD_MODELS] = ",".join(self.preload_models)
        elif ENV_PRELOAD_MODELS in os.environ:
            del os.environ[ENV_PRELOAD_MODELS]

    @classmethod
    def from_env_vars(cls) -> AppStateConfig:
        """Deserialize configuration from environment variables."""
        device = os.environ.get(ENV_DEVICE, "cpu")
        models_dir = os.environ.get(ENV_MODELS_DIR)
        model_filter_str = os.environ.get(ENV_MODEL_FILTER)
        model_filter = [m.strip() for m in model_filter_str.split(",") if m.strip()] if model_filter_str else None
        preload_str = os.environ.get(ENV_PRELOAD_MODELS)
        preload_models = [m.strip() for m in preload_str.split(",") if m.strip()] if preload_str else None

        return cls(
            models_dir=models_dir,
            device=device,
            model_filter=model_filter,
            preload_models=preload_models,
        )
