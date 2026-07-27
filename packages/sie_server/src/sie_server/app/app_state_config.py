from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Environment variable names for configuration
ENV_DEVICE = "SIE_DEVICE"
ENV_DEVICES = "SIE_DEVICES"
ENV_MODELS_DIR = "SIE_MODELS_DIR"
ENV_MODEL_FILTER = "SIE_MODEL_FILTER"
ENV_PRELOAD_MODELS = "SIE_PRELOAD_MODELS"
ENV_PINNED_MODELS = "SIE_PINNED_MODELS"
ENV_POOL = "SIE_POOL"


@dataclass
class AppStateConfig:
    """Configuration for FastAPI app state, passed to lifespan.

    This dataclass holds all startup configuration needed by the lifespan
    context manager to initialize the model registry and related components.
    """

    models_dir: Path | str | None = None
    """Path to models directory (local path, s3://, gs://, abfs://, or abfss://). If None, registry starts empty."""

    device: str = "cpu"
    """Device to load models on (e.g., "cuda:0", "cpu", "mps")."""

    devices: list[str] | None = None
    """Optional device list for whole-model placement (e.g., ["cuda:0", "cuda:1"])."""

    model_filter: list[str] | None = None
    """Optional list of model names to include. If None, all models are available."""

    preload_models: list[str] | None = None
    """Optional list of model names to eagerly load at startup. If None, all models are lazy-loaded."""

    pinned_models: list[str] | None = None
    """Optional list of model names to pin in memory (eager-loaded and never LRU-evicted). If None, no models are pinned."""

    pool_name: str | None = None
    """Optional worker pool identity used by ModelRegistry pool-isolation checks."""

    def __post_init__(self) -> None:
        """Keep scalar and concrete device settings in the same device family."""
        if not self.devices:
            return

        devices = [d.strip() for d in self.devices if d.strip()]
        if not devices:
            self.devices = None
            return

        self.devices = devices
        device_families = {d.split(":", maxsplit=1)[0].lower() for d in devices}
        configured_family = self.device.strip().split(":", maxsplit=1)[0].lower()
        families = ", ".join(sorted(device_families))

        if len(device_families) != 1:
            raise ValueError(
                f"SIE_DEVICE family '{configured_family}' must match SIE_DEVICES families; got multiple: {families}"
            )

        if configured_family == "cpu" and len(device_families) == 1:
            self.device = next(iter(device_families))
            return

        if configured_family not in device_families:
            raise ValueError(f"SIE_DEVICE family '{configured_family}' must match SIE_DEVICES families: {families}")

    def save_to_env_vars(self) -> None:
        """Serialize configuration to environment variables for uvicorn reload mode."""
        if self.models_dir:
            os.environ[ENV_MODELS_DIR] = str(self.models_dir)
        elif ENV_MODELS_DIR in os.environ:
            del os.environ[ENV_MODELS_DIR]

        os.environ[ENV_DEVICE] = self.device

        if self.devices:
            os.environ[ENV_DEVICES] = ",".join(self.devices)
        elif ENV_DEVICES in os.environ:
            del os.environ[ENV_DEVICES]

        # `is not None`: an empty filter must round-trip as an empty env var,
        # not vanish — unset means "no filter" and would re-advertise the full
        # catalog after the uvicorn factory rebuilds state from env.
        if self.model_filter is not None:
            os.environ[ENV_MODEL_FILTER] = ",".join(self.model_filter)
        elif ENV_MODEL_FILTER in os.environ:
            del os.environ[ENV_MODEL_FILTER]

        if self.preload_models:
            os.environ[ENV_PRELOAD_MODELS] = ",".join(self.preload_models)
        elif ENV_PRELOAD_MODELS in os.environ:
            del os.environ[ENV_PRELOAD_MODELS]

        if self.pinned_models:
            os.environ[ENV_PINNED_MODELS] = ",".join(self.pinned_models)
        elif ENV_PINNED_MODELS in os.environ:
            del os.environ[ENV_PINNED_MODELS]

        if self.pool_name:
            os.environ[ENV_POOL] = self.pool_name
        elif ENV_POOL in os.environ:
            del os.environ[ENV_POOL]

    @classmethod
    def from_env_vars(cls) -> AppStateConfig:
        """Deserialize configuration from environment variables."""
        device = os.environ.get(ENV_DEVICE, "cpu")
        devices_str = os.environ.get(ENV_DEVICES)
        devices = [d.strip() for d in devices_str.split(",") if d.strip()] if devices_str else None
        models_dir = os.environ.get(ENV_MODELS_DIR)
        model_filter_str = os.environ.get(ENV_MODEL_FILTER)
        # Unset -> None (no filter); set-but-empty -> [] (zero models).
        model_filter = (
            None if model_filter_str is None else [m.strip() for m in model_filter_str.split(",") if m.strip()]
        )
        preload_str = os.environ.get(ENV_PRELOAD_MODELS)
        preload_models = [m.strip() for m in preload_str.split(",") if m.strip()] if preload_str else None
        pinned_str = os.environ.get(ENV_PINNED_MODELS)
        pinned_models = [m.strip() for m in pinned_str.split(",") if m.strip()] if pinned_str else None
        pool_name = os.environ.get(ENV_POOL) or None

        return cls(
            models_dir=models_dir,
            device=device,
            devices=devices,
            model_filter=model_filter,
            preload_models=preload_models,
            pinned_models=pinned_models,
            pool_name=pool_name,
        )
