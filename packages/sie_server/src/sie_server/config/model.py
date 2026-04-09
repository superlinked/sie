import threading
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from sie_server.config.engine import ComputePrecision

OutputType = Literal["dense", "sparse", "multivector", "score", "json"]
PoolingStrategy = Literal["cls", "mean", "last_token", "splade", "none"]

_MODALITY_NAMES = ("text", "image", "audio", "video")


class InputModalities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: bool = True
    image: bool = False
    audio: bool = False
    video: bool = False

    def to_list(self) -> list[str]:
        return [k for k in _MODALITY_NAMES if getattr(self, k)]


class EmbeddingDim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dim: int


class EncodeTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense: EmbeddingDim | None = None
    sparse: EmbeddingDim | None = None
    multivector: EmbeddingDim | None = None


class ScoreTask(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractTask(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tasks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encode: EncodeTask | None = None
    score: ScoreTask | None = None
    extract: ExtractTask | None = None


class AdapterOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loadtime: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class ProfileAdaptiveBatching(BaseModel):
    """Per-model adaptive batching overrides.

    All fields are optional. None means inherit from engine config or parent
    profile. This enables fieldwise merge: a child profile can override one
    field while inheriting the rest from the parent or engine defaults.
    """

    model_config = ConfigDict(extra="forbid")

    target_p50_ms: float | None = None
    calibration_multiplier: float | None = None
    min_target_p50_ms: float | None = None
    max_target_p50_ms: float | None = None
    min_wait_ms: float | None = None
    max_wait_ms: float | None = None
    gain: float | None = None
    integral_gain: float | None = None


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extends: str | None = None
    max_batch_tokens: int | None = None
    compute_precision: ComputePrecision | None = None
    adapter_path: str | None = None
    adapter_options: AdapterOptions = AdapterOptions()
    adaptive_batching: ProfileAdaptiveBatching | None = None


class ResolvedProfile(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    max_batch_tokens: int
    compute_precision: ComputePrecision | None
    adapter_path: str
    loadtime: MappingProxyType[str, Any]
    runtime: MappingProxyType[str, Any]
    adaptive_batching: ProfileAdaptiveBatching | None = None


def _merge_profile_adaptive_batching(
    parent: ProfileAdaptiveBatching | None,
    child: ProfileAdaptiveBatching | None,
) -> ProfileAdaptiveBatching | None:
    """Merge child adaptive batching overrides onto parent, fieldwise.

    None fields in child inherit from parent. If both are None, returns None.
    """
    if parent is None and child is None:
        return None
    if parent is None:
        return child
    if child is None:
        return parent

    # Fieldwise merge: child overrides parent per-field
    return ProfileAdaptiveBatching(
        target_p50_ms=child.target_p50_ms if child.target_p50_ms is not None else parent.target_p50_ms,
        calibration_multiplier=child.calibration_multiplier
        if child.calibration_multiplier is not None
        else parent.calibration_multiplier,
        min_target_p50_ms=child.min_target_p50_ms if child.min_target_p50_ms is not None else parent.min_target_p50_ms,
        max_target_p50_ms=child.max_target_p50_ms if child.max_target_p50_ms is not None else parent.max_target_p50_ms,
        min_wait_ms=child.min_wait_ms if child.min_wait_ms is not None else parent.min_wait_ms,
        max_wait_ms=child.max_wait_ms if child.max_wait_ms is not None else parent.max_wait_ms,
        gain=child.gain if child.gain is not None else parent.gain,
        integral_gain=child.integral_gain if child.integral_gain is not None else parent.integral_gain,
    )


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Intentionally non-serializable; rebuilt on demand after deserialization.
    _resolved_cache: dict[str, ResolvedProfile] = PrivateAttr(default_factory=dict)
    _resolved_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    sie_id: str
    hf_id: str | None = None
    hf_revision: str | None = None
    weights_path: Path | None = None
    inputs: InputModalities = InputModalities()
    tasks: Tasks
    max_sequence_length: int | None = None
    profiles: dict[str, ProfileConfig]

    @model_validator(mode="after")
    def validate_weight_source(self) -> "ModelConfig":
        if self.hf_id is None and self.weights_path is None:
            msg = "At least one of 'hf_id' or 'weights_path' must be set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_profiles(self) -> "ModelConfig":
        if "default" not in self.profiles:
            msg = "'default' key must exist in profiles"
            raise ValueError(msg)
        for name, profile in self.profiles.items():
            if profile.extends is not None:
                if profile.extends not in self.profiles:
                    msg = f"Profile '{name}' extends unknown profile '{profile.extends}'"
                    raise ValueError(msg)
                parent = self.profiles[profile.extends]
                if parent.extends is not None:
                    msg = f"Profile chaining is not allowed: '{name}' -> '{profile.extends}' -> '{parent.extends}'"
                    raise ValueError(msg)
            else:
                if profile.adapter_path is None:
                    msg = f"Profile '{name}' must have 'adapter_path' set (or use 'extends')"
                    raise ValueError(msg)
                if profile.max_batch_tokens is None:
                    msg = f"Profile '{name}' must have 'max_batch_tokens' set (or use 'extends')"
                    raise ValueError(msg)
        return self

    def resolve_profile(self, name: str) -> ResolvedProfile:
        if name in self._resolved_cache:
            return self._resolved_cache[name]
        with self._resolved_lock:
            # Double-check after acquiring lock
            if name in self._resolved_cache:
                return self._resolved_cache[name]
            resolved = self._resolve_profile_uncached(name)
            self._resolved_cache[name] = resolved
            return resolved

    def _resolve_profile_uncached(self, name: str) -> ResolvedProfile:
        if name not in self.profiles:
            msg = f"Profile '{name}' not found. Available: {list(self.profiles.keys())}"
            raise ValueError(msg)

        profile = self.profiles[name]

        if profile.extends is None:
            # Validators guarantee adapter_path and max_batch_tokens are set
            # for non-extending profiles.
            if profile.adapter_path is None:
                msg = f"Profile '{name}': adapter_path must be set"
                raise ValueError(msg)
            if profile.max_batch_tokens is None:
                msg = f"Profile '{name}': max_batch_tokens must be set"
                raise ValueError(msg)
            return ResolvedProfile(
                max_batch_tokens=profile.max_batch_tokens,
                compute_precision=profile.compute_precision,
                adapter_path=profile.adapter_path,
                loadtime=MappingProxyType(dict(profile.adapter_options.loadtime)),
                runtime=MappingProxyType(dict(profile.adapter_options.runtime)),
                adaptive_batching=profile.adaptive_batching,
            )

        # Resolve via parent — validators guarantee parent exists and has no chaining
        parent_name = profile.extends
        parent = self.profiles[parent_name]

        # Start with parent values
        max_batch_tokens = parent.max_batch_tokens
        compute_precision = parent.compute_precision
        adapter_path = parent.adapter_path
        loadtime = dict(parent.adapter_options.loadtime)
        runtime = dict(parent.adapter_options.runtime)

        # Override with child's non-None top-level fields
        if profile.max_batch_tokens is not None:
            max_batch_tokens = profile.max_batch_tokens
        if profile.compute_precision is not None:
            compute_precision = profile.compute_precision
        if profile.adapter_path is not None:
            adapter_path = profile.adapter_path

        # For adapter_options: full replacement if child specifies non-empty
        if profile.adapter_options.loadtime:
            loadtime = dict(profile.adapter_options.loadtime)
        if profile.adapter_options.runtime:
            runtime = dict(profile.adapter_options.runtime)

        # Adaptive batching: fieldwise merge (child overrides parent per-field)
        adaptive_batching = _merge_profile_adaptive_batching(parent.adaptive_batching, profile.adaptive_batching)

        if max_batch_tokens is None:
            msg = f"Resolved profile '{name}': max_batch_tokens must be set"
            raise ValueError(msg)
        if adapter_path is None:
            msg = f"Resolved profile '{name}': adapter_path must be set"
            raise ValueError(msg)

        return ResolvedProfile(
            max_batch_tokens=max_batch_tokens,
            compute_precision=compute_precision,
            adapter_path=adapter_path,
            loadtime=MappingProxyType(loadtime),
            runtime=MappingProxyType(runtime),
            adaptive_batching=adaptive_batching,
        )

    @property
    def name(self) -> str:
        return self.sie_id

    @property
    def outputs(self) -> list[str]:
        result: list[str] = []
        encode = self.tasks.encode
        if encode is not None:
            if encode.dense is not None:
                result.append("dense")
            if encode.sparse is not None:
                result.append("sparse")
            if encode.multivector is not None:
                result.append("multivector")
        if self.tasks.score is not None:
            result.append("score")
        if self.tasks.extract is not None:
            result.append("json")
        return result

    @property
    def dims(self) -> dict[str, int]:
        result: dict[str, int] = {}
        encode = self.tasks.encode
        if encode is not None:
            if encode.dense is not None:
                result["dense"] = encode.dense.dim
            if encode.sparse is not None:
                result["sparse"] = encode.sparse.dim
            if encode.multivector is not None:
                result["multivector"] = encode.multivector.dim
        return result
