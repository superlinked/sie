"""Engine configuration for SIE Server.

Defines the EngineConfig Pydantic model that controls server-wide settings
like batching, memory management, and performance tuning.

See DESIGN.md Section 10.1 for full specification.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Attention backend options
AttentionBackend = Literal["auto", "flash_attention_2", "sdpa", "eager"]

# Compute precision options (how model runs on GPU)
ComputePrecision = Literal["float16", "bfloat16", "float32"]


class EngineConfig(BaseSettings):
    """Engine configuration loaded from engine.yaml or environment variables.

    Environment variables are prefixed with SIE_ and use uppercase names.
    Example: SIE_MAX_BATCH_REQUESTS=128

    Note: max_batch_tokens is per-model (in model config), not engine-level.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIE_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # Batching (max_batch_tokens is per-model in model config)
    max_batch_requests: Annotated[
        int,
        Field(description="Maximum requests per batch"),
    ] = 64
    max_batch_wait_ms: Annotated[
        int,
        Field(description="Maximum milliseconds to wait for batch to fill"),
    ] = 10
    max_concurrent_requests: Annotated[
        int,
        Field(description="Maximum concurrent requests (queue size)"),
    ] = 512

    # Memory
    memory_pressure_threshold_percent: Annotated[
        int,
        Field(ge=50, le=99, description="VRAM usage percent that triggers LRU eviction"),
    ] = 85

    # Disk cache
    disk_cache_enabled: Annotated[
        bool,
        Field(description="Enable LRU disk cache management"),
    ] = True
    disk_pressure_threshold_percent: Annotated[
        int,
        Field(
            ge=50,
            le=99,
            description="Disk usage percent that triggers LRU eviction before model download",
        ),
    ] = 85

    # LoRA
    max_loras_per_model: Annotated[
        int,
        Field(
            ge=1,
            description="Maximum number of LoRA adapters to keep loaded per model. "
            "LRU eviction when limit is reached. Can be overridden per-model via adapter_options_loadtime.",
        ),
    ] = 10

    # Performance
    preprocessor_workers: Annotated[
        int,
        Field(ge=1, description="Number of preprocessing worker threads"),
    ] = 4
    attention_backend: Annotated[
        AttentionBackend,
        Field(description="Attention implementation: auto, flash_attention_2, sdpa, eager"),
    ] = "auto"
    default_compute_precision: Annotated[
        ComputePrecision,
        Field(description="Default compute precision for models: float16, bfloat16, float32"),
    ] = "float16"
    instrumentation: Annotated[
        bool,
        Field(description="Enable detailed batch instrumentation for debugging"),
    ] = False

    # Paths
    models_dir: Annotated[
        Path,
        Field(description="Directory containing model configs"),
    ] = Path("./models")
