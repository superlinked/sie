"""Engine configuration for SIE Server.

Defines the EngineConfig Pydantic model that controls server-wide settings
like batching, memory management, and performance tuning.

See DESIGN.md Section 10.1 for full specification.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from sie_server.core.oom import OomRecoveryConfig

# Attention backend options
AttentionBackend = Literal["auto", "flash_attention_2", "sdpa", "eager"]

# Compute precision options (how model runs on GPU)
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Recovery action names exposed to YAML / env. Mirror ``OomRecoveryAction``.
OomRecoveryActionName = Literal["cache_clear", "evict_lru", "split_batch"]


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


class OomRecoveryConfigPydantic(BaseModel):
    """Worker-side OOM recovery settings.

    Wraps :class:`sie_server.core.oom.OomRecoveryConfig` for YAML / env
    parsing. Use ``to_runtime()`` to obtain the frozen dataclass that the
    worker actually consumes.

    Env example: ``SIE_OOM_RECOVERY__ENABLED=false`` disables recovery.
    """

    enabled: Annotated[
        bool,
        Field(description="Master switch for reactive OOM recovery in the worker."),
    ] = True
    strategy: Annotated[
        list[OomRecoveryActionName],
        Field(
            min_length=1,
            description=(
                "Ordered list of recovery actions. cache_clear is cheap, evict_lru "
                "frees a sibling model, split_batch is recursive and terminal. "
                "Must be non-empty; duplicates are deduplicated (preserving order)."
            ),
        ),
    ] = ["cache_clear", "evict_lru", "split_batch"]

    @model_validator(mode="after")
    def _dedup_strategy_preserve_order(self) -> "OomRecoveryConfigPydantic":
        """Drop duplicates in ``strategy`` while preserving first-occurrence order.

        Duplicate strategies are mostly harmless (a second ``cache_clear``
        is wasted work; a second ``evict_lru`` would evict another sibling
        model that the operator probably didn't intend to lose). Silently
        deduping is friendlier than rejecting at parse time, since YAML
        composition / overlays can accidentally produce duplicates.
        """
        seen: set[str] = set()
        deduped: list[OomRecoveryActionName] = []
        for action in self.strategy:
            if action not in seen:
                seen.add(action)
                deduped.append(action)
        if len(deduped) != len(self.strategy):
            self.strategy = deduped
        return self

    max_split_depth: Annotated[
        int,
        Field(
            ge=0,
            le=8,
            description=(
                "Maximum recursion depth for split_batch. Each step halves the "
                "batch; depth=4 permits up to 16 sub-batches."
            ),
        ),
    ] = 4
    eviction_lock_timeout_s: Annotated[
        float,
        Field(
            ge=0.1,
            le=60.0,
            description=(
                "How long to wait for the registry's load-lock during evict_lru "
                "before giving up and trying the next strategy."
            ),
        ),
    ] = 5.0
    retry_after_s: Annotated[
        int,
        Field(
            ge=1,
            le=60,
            description="Retry-After header value when a request fails with RESOURCE_EXHAUSTED.",
        ),
    ] = 5

    def to_runtime(self) -> "OomRecoveryConfig":
        """Convert to the frozen dataclass consumed by ``BatchExecutor``."""
        # Imported lazily to avoid a circular import: ``core/__init__`` pulls
        # in ``loader`` which already depends on this module.
        from sie_server.core.oom import OomRecoveryAction, OomRecoveryConfig

        return OomRecoveryConfig(
            enabled=self.enabled,
            strategy=tuple(OomRecoveryAction(name) for name in self.strategy),
            max_split_depth=self.max_split_depth,
            eviction_lock_timeout_s=self.eviction_lock_timeout_s,
            retry_after_s=self.retry_after_s,
        )


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
    idle_evict_s: Annotated[
        int | None,
        Field(
            ge=10,
            le=86400,
            description=(
                "Unload models that have been idle (no requests) for this many seconds. "
                "Additive to the 85% pressure monitor: catches cold models before they "
                "build up and become eviction candidates under load. "
                "None disables (default); set ``SIE_IDLE_EVICT_S=300`` for a 5-minute TTL."
            ),
        ),
    ] = None

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

    # Reactive OOM recovery
    oom_recovery: Annotated[
        OomRecoveryConfigPydantic,
        Field(
            description=(
                "Reactive OOM recovery in the worker dispatch path. Default "
                "is enabled: on OOM, try cache_clear, then evict_lru, then "
                "recursively halve the batch. Disable with "
                "SIE_OOM_RECOVERY__ENABLED=false (or the convenience flag "
                "SIE_DISABLE_OOM_RECOVERY=1) for incident triage."
            ),
        ),
    ] = OomRecoveryConfigPydantic()

    # Paths
    models_dir: Annotated[
        Path,
        Field(description="Directory containing model configs"),
    ] = Path("./models")

    @model_validator(mode="after")
    def _apply_oom_kill_switch(self) -> "EngineConfig":
        """Honour ``SIE_DISABLE_OOM_RECOVERY=1`` as a top-level kill switch.

        Lives here (not on ``OomRecoveryConfigPydantic``) so the convenience
        env var can be flat — operators don't have to learn the nested
        delimiter syntax during an incident. ``SIE_DISABLE_OOM_RECOVERY``
        wins over an explicit ``SIE_OOM_RECOVERY__ENABLED=true`` because
        this validator runs *after* the nested settings parse, and we
        intentionally treat it as the most disruptive operator override.

        Note on validation semantics: ``model_copy(update=...)`` performs a
        structural mutation that **bypasses field validators**. Safe here
        because ``enabled`` is a plain ``bool`` with no constraints, but
        future contributors mutating constrained fields this way should
        prefer ``model_validate({**dump(), "field": value})`` to re-run
        validators.
        """
        flag = os.environ.get("SIE_DISABLE_OOM_RECOVERY", "").lower()
        if flag in ("1", "true", "yes") and self.oom_recovery.enabled:
            self.oom_recovery = self.oom_recovery.model_copy(update={"enabled": False})
        return self
