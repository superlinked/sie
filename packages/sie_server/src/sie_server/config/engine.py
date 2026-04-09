"""Engine configuration for SIE Server.

Defines the EngineConfig Pydantic model that controls server-wide settings
like batching, memory management, and performance tuning.

See DESIGN.md Section 10.1 for full specification.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Attention backend options
AttentionBackend = Literal["auto", "flash_attention_2", "sdpa", "eager"]

# Compute precision options (how model runs on GPU)
ComputePrecision = Literal["float16", "bfloat16", "float32"]


class AdaptiveBatchingConfig(BaseModel):
    """Configuration for adaptive batch control.

    When enabled, the server dynamically adjusts ``max_batch_wait_ms`` and
    ``max_batch_cost`` (token limit) per model based on observed p50 latency
    and GPU saturation (batch fill ratio).

    The latency target (``target_p50_ms``) can be set explicitly or left as
    ``null`` for auto-calibration. When null, the controller measures
    inference-only p50 during the first N requests and derives the target
    as ``inference_p50 × calibration_multiplier``.
    """

    enabled: Annotated[
        bool,
        Field(description="Enable adaptive batch wait control"),
    ] = True
    target_p50_ms: Annotated[
        float | None,
        Field(
            gt=0,
            description="Latency SLO: desired p50 in milliseconds. "
            "null = auto-calibrate from observed inference latency.",
        ),
    ] = None
    calibration_multiplier: Annotated[
        float,
        Field(
            gt=1,
            description="Auto-calibration: target = inference_p50 * multiplier",
        ),
    ] = 1.5
    min_target_p50_ms: Annotated[
        float,
        Field(ge=1, description="Floor for auto-calibrated target"),
    ] = 5.0
    max_target_p50_ms: Annotated[
        float,
        Field(ge=10, description="Ceiling for auto-calibrated target"),
    ] = 500.0
    min_wait_ms: Annotated[
        float,
        Field(ge=0.1, description="Minimum batch wait time in milliseconds"),
    ] = 1.0
    max_wait_ms: Annotated[
        float,
        Field(ge=1, description="Maximum batch wait time in milliseconds"),
    ] = 50.0
    gain: Annotated[
        float,
        Field(gt=0, le=1, description="Proportional controller gain (0.1=slow, 0.5=aggressive)"),
    ] = 0.3
    integral_gain: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Integral controller gain. 0 = proportional-only.",
        ),
    ] = 0.05
    window_size: Annotated[
        int,
        Field(ge=10, description="Rolling latency sample window size"),
    ] = 200
    update_interval: Annotated[
        int,
        Field(ge=1, description="Batches between controller updates"),
    ] = 10

    @model_validator(mode="after")
    def validate_invariants(self) -> "AdaptiveBatchingConfig":
        """Check cross-field invariants."""
        if self.min_wait_ms > self.max_wait_ms:
            msg = f"min_wait_ms ({self.min_wait_ms}) must be <= max_wait_ms ({self.max_wait_ms})"
            raise ValueError(msg)
        if self.min_target_p50_ms > self.max_target_p50_ms:
            msg = (
                f"min_target_p50_ms ({self.min_target_p50_ms}) must be <= max_target_p50_ms ({self.max_target_p50_ms})"
            )
            raise ValueError(msg)
        return self


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
        float,
        Field(description="Maximum milliseconds to wait for batch to fill"),
    ] = 10.0
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

    # Adaptive batching
    adaptive_batching: Annotated[
        AdaptiveBatchingConfig,
        Field(description="Adaptive batch wait controller settings"),
    ] = AdaptiveBatchingConfig()

    # Paths
    models_dir: Annotated[
        Path,
        Field(description="Directory containing model configs"),
    ] = Path("./models")
