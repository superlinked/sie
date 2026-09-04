import logging
import re
import threading
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from sie_server.config.engine import ComputePrecision
from sie_server.config.package_artifacts import (
    PackageArtifactDeclaration,
    has_package_artifact_declaration,
    parse_package_artifact_declaration,
)
from sie_server.config.serving_artifacts import (
    ServingArtifactDeclaration,
    parse_serving_artifact_declaration,
)

logger = logging.getLogger(__name__)

OutputType = Literal["dense", "sparse", "multivector", "score", "json", "tokens"]
PoolingStrategy = Literal["cls", "mean", "last_token", "splade", "none"]

_MODALITY_NAMES = ("text", "image", "audio", "video", "document")
_MAX_POOL_NAME_LEN = 128
_CHAT_TEMPLATE_KWARGS = frozenset({"enable_thinking", "guardian_config"})
_GUARDIAN_CONFIG_KWARGS = frozenset({"risk_name"})
_MAX_GUARDIAN_RISK_NAME_LEN = 128
_SERVING_ARTIFACT_PRECISION: dict[str, ComputePrecision] = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "int8_bfloat16": "bfloat16",
    "int8_float16": "float16",
    "int8_float32": "float32",
}


def validate_chat_template_kwargs(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate the bounded operator-owned tokenizer template contract.

    These values cross from model YAML into ``apply_chat_template(**kwargs)``.
    Keeping the accepted keys and nested shapes explicit prevents a catalog
    edit from silently enabling a tokenizer-specific extension with arbitrary
    behavior or unbounded data.
    """
    if value is None:
        return None

    unsupported = set(value) - _CHAT_TEMPLATE_KWARGS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported chat_template_kwargs key(s): {names}")

    enable_thinking = value.get("enable_thinking")
    if "enable_thinking" in value and not isinstance(enable_thinking, bool):
        raise ValueError("chat_template_kwargs.enable_thinking must be a boolean")

    guardian_config = value.get("guardian_config")
    if "guardian_config" in value:
        if not isinstance(guardian_config, dict):
            raise ValueError("chat_template_kwargs.guardian_config must be an object")
        unsupported_guardian = set(guardian_config) - _GUARDIAN_CONFIG_KWARGS
        if unsupported_guardian:
            names = ", ".join(sorted(unsupported_guardian))
            raise ValueError(f"unsupported chat_template_kwargs.guardian_config key(s): {names}")
        risk_name = guardian_config.get("risk_name")
        if not isinstance(risk_name, str) or not risk_name or len(risk_name) > _MAX_GUARDIAN_RISK_NAME_LEN:
            raise ValueError(
                "chat_template_kwargs.guardian_config.risk_name must be a non-empty string "
                f"of at most {_MAX_GUARDIAN_RISK_NAME_LEN} characters"
            )

    return value


# Served-model version identity. An *immutable* HF revision is a full 40-char git
# commit SHA (SHA-1, lowercase hex). Branch/tag names ("main", "v1.0") resolve to
# *moving* targets on the Hub, so they are NOT acceptable pins for a promoted/served
# model — the immutable-id contract is that a given ``sie_id`` maps to identical
# weights forever. This regex is the single authoritative rule for both the base
# weights (``hf_revision``) and pinned LoRA refs (``loadtime.lora_paths`` dict values,
# #2113); staging/deploy tooling mirrors it rather than importing this module.
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def is_immutable_revision(revision: str | None) -> bool:
    """Return True when ``revision`` is a full 40-char git commit SHA (immutable)."""
    return revision is not None and _IMMUTABLE_REVISION_RE.match(revision) is not None


def lora_entry_ref(value: Any) -> tuple[str, str | None]:
    """Normalize one ``loadtime.lora_paths`` dict-form value to ``(id, revision)``.

    The dict form's values are either a bare id string (unpinned, ``revision``
    is ``None``) or a ``{id, revision}`` mapping (#2113). The list form and the
    legacy scalar ``runtime.lora_id`` are bare-only and never reach this helper
    with a mapping.
    """
    if isinstance(value, Mapping):
        return str(value.get("id")), value.get("revision")
    return str(value), None


class InputModalities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: bool = True
    image: bool = False
    audio: bool = False
    video: bool = False
    document: bool = False

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


class GenerateCapabilities(BaseModel):
    """Generation capability flags advertised by the model config.

    Gateway-readable surface; used by ``proxy_generate`` to enforce that a
    requested grammar / tools / streaming flavour is actually supported.
    ``grammar`` accepts ``json_schema``, ``regex``, and ``ebnf``; the
    capability gate at the gateway uses this list to reject unsupported
    kinds before any work hits the queue.
    """

    model_config = ConfigDict(extra="forbid")

    grammar: list[Literal["json_schema", "regex", "ebnf"]] = []
    streaming: bool = True
    tools: bool = False
    # Advertises that the model is validated for code generation (HumanEval /
    # MBPP pass@1 in the generation quality gate). Informational only — unlike
    # ``grammar``, the gateway does not reject requests on this flag; it lets
    # the SDK/UI surface the capability and backs the ``model="code"`` alias.
    code: bool = False
    # Same, for text-to-SQL (Spider execution accuracy); backs ``model="sql"``.
    sql: bool = False
    # Same, for the content-moderation / policy-check job (CHECK POLICY): a
    # generative guard model that emits a safe/unsafe verdict (e.g. Granite
    # Guardian, measured on ToxicChat). Backs the ``model="guard"`` alias.
    guard: bool = False


# Kinds permitted in ``prewarm_grammars`` entries. Mirrors the
# capability list :class:`GenerateCapabilities` advertises for the
# request path so an operator cannot prewarm a kind the worker would
# refuse to serve. Same set as :class:`GenerateCapabilities.grammar`
# (the literal in :data:`GrammarKind`) since EBNF prewarm is just as
# valid as runtime EBNF compile.
PrewarmGrammarKind = Literal["json_schema", "regex", "ebnf"]


class PrewarmGrammar(BaseModel):
    """Operator-declared grammar to compile during model load.

    Pre-compiling hot schemas/regexes at worker boot moves Outlines compile
    cost out of cold-start TTFT. Each entry corresponds to one ``(kind, value)``
    pair that would otherwise be compiled lazily on first request.

    ``name`` is a human-readable label used in log lines and is otherwise
    informational — the cache key is derived from ``value`` via
    :func:`~sie_server.types.grammar.hash_grammar`. ``kind`` must be one
    of :data:`PrewarmGrammarKind` (the narrower per-capability surface,
    not the full :data:`GrammarKind` literal). ``value`` matches the
    on-wire shape: a JSON Schema ``dict`` for ``kind: json_schema`` and
    a regex/EBNF ``str`` for ``kind: regex`` / ``kind: ebnf``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: PrewarmGrammarKind
    value: dict[str, Any] | str

    @model_validator(mode="after")
    def validate_value_shape(self) -> "PrewarmGrammar":
        """Cross-field check: value type matches kind discriminator.

        Defence-in-depth — pydantic's union accepts either shape at parse
        time but a regex with a dict value (or vice versa) would only
        surface as a compile failure later. Reject loudly at config-load.
        """
        if self.kind == "json_schema" and not isinstance(self.value, dict):
            msg = f"prewarm grammar '{self.name}': kind=json_schema requires a dict value, got {type(self.value).__name__}"
            raise ValueError(msg)
        if self.kind == "regex" and not isinstance(self.value, str):
            msg = f"prewarm grammar '{self.name}': kind=regex requires a str value, got {type(self.value).__name__}"
            raise ValueError(msg)
        if self.kind == "ebnf" and not isinstance(self.value, str):
            msg = f"prewarm grammar '{self.name}': kind=ebnf requires a str value, got {type(self.value).__name__}"
            raise ValueError(msg)
        return self


class GenerateTask(BaseModel):
    """Generation task declaration.

    ``context_length`` is the maximum shared prompt-plus-completion envelope by
    default. Encoder-decoder adapters may instead declare independent axes, in
    which case it bounds encoder input and ``max_output_tokens`` independently
    caps decoder output. The gateway uses ``max_output_tokens`` for
    pre-admission, and the worker authoritatively enforces it as the per-request
    hard cap on ``max_new_tokens`` before adapter dispatch.

    ``chat_template_kwargs`` are a bounded operator-owned mapping forwarded
    to the tokenizer's ``apply_chat_template(**kwargs)`` call when the worker
    renders an OpenAI-shaped ``messages`` request. The accepted schema covers
    ``enable_thinking`` and Granite Guardian's bounded ``guardian_config``;
    unknown keys and invalid nested shapes fail at config load. Empty dict by
    default — non-chat / prompt-shape requests ignore the field.

    ``prewarm_grammars`` is an optional list of grammars to compile at
    model-load time so the cold-start TTFT for these schemas excludes
    Outlines compile cost. See :class:`PrewarmGrammar` for the entry
    shape; the worker iterates the list once on boot and silently
    continues past individual compile failures (which are surfaced via
    ``sie.worker.generation.grammar.compile.duration`` with
    ``phase="prewarm", outcome="error"``).

    ``kv_budget_tokens`` for admission control lives on
    :class:`ProfileConfig` rather than here, because the
    budget is a per-worker/per-profile shape rather than a per-task
    semantic.

    ``grammar_profile`` optionally names a profile that grammar-constrained
    requests (OpenAI ``response_format`` / SIE-native ``grammar``) must run on.
    When set, the gateway rewrites such a request's model id to the
    ``{sie_id}:{grammar_profile}`` variant so it is served by that profile,
    while unconstrained requests keep the request's resolved profile. A
    profile may override this with its own grammar-safe sibling so context,
    hardware launch shape, and thinking mode remain unchanged. A directly
    inheriting explicit variant may remain selected when its resolved launch
    settings are already grammar-safe. This exists
    because some throughput optimisations are incompatible with
    decode-time grammar enforcement — notably NEXTN/MTP speculative decoding
    bypasses SGLang's Outlines FSM (leaks out-of-schema keys, truncates
    mid-JSON), so a model whose default profile is speculative points
    ``grammar_profile`` at a non-speculative profile (e.g. ``no-spec``). ``None``
    (default) means no rewrite — grammar runs on the request's resolved profile.
    """

    model_config = ConfigDict(extra="forbid")

    context_length: int
    max_output_tokens: int
    capabilities: GenerateCapabilities = GenerateCapabilities()
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    prewarm_grammars: list[PrewarmGrammar] = Field(default_factory=list)
    grammar_profile: str | None = None

    @field_validator("chat_template_kwargs")
    @classmethod
    def validate_chat_template_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_chat_template_kwargs(value) or {}


class Tasks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encode: EncodeTask | None = None
    score: ScoreTask | None = None
    extract: ExtractTask | None = None
    generate: GenerateTask | None = None


class AdapterOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loadtime: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lora_paths(self) -> "AdapterOptions":
        """Validate the ``loadtime.lora_paths`` LoRA-ref spellings (#2113).

        The dict form (served-name -> ref) is the pinnable spelling: a value is
        either a bare id string (unpinned, resolves at the Hub's default branch)
        or ``{id: <repo>, revision: <40-hex SHA>}`` — the same immutability rule
        as ``hf_revision``, so a served id cannot silently drift when the
        adapter repo moves. The list form and the legacy scalar
        ``runtime.lora_id`` stay bare-only; other ``lora_paths`` shapes are left
        for the loader's existing warn-and-ignore path.
        """
        lora_paths = self.loadtime.get("lora_paths")
        if isinstance(lora_paths, Mapping):
            for served, value in lora_paths.items():
                if not isinstance(value, Mapping):
                    continue
                unknown = set(value) - {"id", "revision"}
                if unknown:
                    msg = (
                        f"loadtime.lora_paths[{served!r}] has unknown key(s) {sorted(unknown)!r}; "
                        "a pinned LoRA ref is exactly {id: <hf repo>, revision: <40-hex commit SHA>}"
                    )
                    raise ValueError(msg)
                ref_id = value.get("id")
                if not isinstance(ref_id, str) or not ref_id:
                    msg = f"loadtime.lora_paths[{served!r}] must set 'id' to a non-empty repo id string, got {ref_id!r}"
                    raise ValueError(msg)
                revision = value.get("revision")
                if revision is not None and not (isinstance(revision, str) and is_immutable_revision(revision)):
                    msg = (
                        f"loadtime.lora_paths[{served!r}] pins revision={revision!r}, which is not an "
                        "immutable 40-char commit SHA — a branch/tag name (e.g. 'main') drifts on the "
                        "Hub. Pin the resolved commit SHA instead, or omit 'revision'."
                    )
                    raise ValueError(msg)
        elif isinstance(lora_paths, (list, tuple)):
            for value in lora_paths:
                if isinstance(value, Mapping):
                    msg = (
                        "loadtime.lora_paths list entries must be bare id strings; the pinned "
                        "{id, revision} spelling is only valid in the served-name -> ref dict form"
                    )
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_speculative_draft_model(self) -> "AdapterOptions":
        """Validate the optional immutable speculative-draft checkpoint pin."""
        speculative = self.loadtime.get("speculative")
        if not isinstance(speculative, Mapping):
            return self

        revision = speculative.get("draft_model_revision")
        if revision is None:
            return self
        draft_model = speculative.get("draft_model")
        if not isinstance(draft_model, str) or not draft_model:
            msg = "loadtime.speculative.draft_model_revision requires a non-empty draft_model repo id"
            raise ValueError(msg)
        if not isinstance(revision, str) or not is_immutable_revision(revision):
            msg = (
                f"loadtime.speculative.draft_model_revision={revision!r} is not an immutable "
                "40-char commit SHA; pin the resolved draft checkpoint commit"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_serving_artifact_shape(self) -> "AdapterOptions":
        # Source identity lives on ModelConfig, so only parse the closed nested
        # declaration and reject loader-owned injection fields at this layer.
        parse_serving_artifact_declaration(self.loadtime)
        return self


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
    """Per-profile configuration.

    ``kv_budget_tokens`` is the per-worker KV-cache budget
    used by the streaming admission controller to reject requests whose
    ``input_tokens_estimate + max_new_tokens`` would push the worker
    over capacity. **Required** (positive int) for profiles whose
    ``adapter_path`` resolves to a ``GenerationAdapter`` subclass —
    i.e. for any profile attached to a model with ``tasks.generate``
    set. Calibration of the actual value lives in the calibration
    follow-up; until then the model YAML may carry a sentinel
    placeholder, but a missing / zero / negative value at config-load
    time is a hard error pointing operators at the calibration
    deliverable.

    ``admission_enabled`` gates admission control per
    profile. ``None`` defers to the ``SIE_GENERATION_ADMISSION`` env
    var (default-off until the calibration ablation flips it); explicit
    ``True`` / ``False`` wins unless the env var is set to ``on`` or
    ``off`` (which override the profile in both directions).

    ``max_output_tokens`` optionally overrides the model-level generation cap
    for the materialized ``model:profile`` variant. This lets a long-context
    reasoning profile expose the checkpoint's supported decode budget without
    weakening the conservative cap on the bare model or shorter profiles.

    ``chat_template_kwargs`` optionally overrides the same bounded model-level
    :class:`GenerateTask` defaults for the materialized ``model:profile``
    variant. This keeps tokenizer render presets such as explicit thinking
    mode on the profile identity while preserving the bare model's safer
    non-thinking default. The profile mapping is merged over the task mapping;
    request-time kwargs are still applied later by the generation processor.

    ``grammar_profile`` optionally overrides the model-level grammar fallback
    for this concrete profile. It must name a non-default sibling profile. The
    gateway uses it only for constrained requests, allowing a long-context or
    thinking profile to route to a non-speculative twin without losing its
    context window, hardware launch shape, or tokenizer mode.
    """

    model_config = ConfigDict(extra="forbid")

    extends: str | None = None
    max_batch_tokens: int | None = None
    compute_precision: ComputePrecision | None = None
    adapter_path: str | None = None
    adapter_options: AdapterOptions = AdapterOptions()
    adaptive_batching: ProfileAdaptiveBatching | None = None
    kv_budget_tokens: int | None = None
    admission_enabled: bool | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    chat_template_kwargs: dict[str, Any] | None = None
    grammar_profile: str | None = None

    @field_validator("chat_template_kwargs")
    @classmethod
    def validate_chat_template_kwargs(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return validate_chat_template_kwargs(value)


class ResolvedProfile(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    max_batch_tokens: int
    compute_precision: ComputePrecision | None
    adapter_path: str
    loadtime: MappingProxyType[str, Any]
    runtime: MappingProxyType[str, Any]
    adaptive_batching: ProfileAdaptiveBatching | None = None
    kv_budget_tokens: int | None = None
    admission_enabled: bool | None = None
    max_output_tokens: int | None = None
    chat_template_kwargs: MappingProxyType[str, Any] | None = None
    grammar_profile: str | None = None


# Coarse per-family KV-bytes-per-token constants used by the derived-budget
# warning. Real calibration lives in the calibration follow-up; these constants
# are intentionally conservative (overestimates) so the warning fires for
# genuinely over-subscribed configurations but stays silent for the calibrated
# values that follow-up will publish. ``None`` skips the warning for unknown
# families.
_KV_BYTES_PER_TOKEN_BY_FAMILY: dict[str, int] = {
    # Qwen3-4B-Instruct: 36 layers × 8 KV heads × 128 head_dim × 2 bytes
    # (bf16) × 2 (K+V) ≈ ~150 KB/token. Round up.
    "qwen3-4b": 160_000,
}

# Coarse GPU capacity assumption for the derived-budget warning. Real
# deployment surfaces this via SGLang's ``mem_fraction_static`` × the
# device's reported total memory; this constant is the fallback used
# when no GPU is available at config-load time (CI, dry-run validation).
_DEFAULT_GPU_MEMORY_GB = 24.0

# Profile-name → GPU memory (GB) mapping. Used by
# :func:`_maybe_warn_oversubscribed_budget` so per-profile entries like
# ``a100-40gb`` and ``h100`` don't false-positive against the L4-baseline
# default of 24 GB. The matcher is substring-based on the profile name
# (lowercased) so variant names like ``a100-80gb`` or ``h100-sxm`` still
# resolve. ``default`` retains the L4 baseline (where the historical
# fallback comes from).
_GPU_MEMORY_GB_BY_PROFILE_HINT: tuple[tuple[str, float], ...] = (
    ("h200", 141.0),
    ("rtx-pro-6000", 96.0),  # RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7)
    ("h100", 80.0),
    ("a100-80", 80.0),
    ("a100-40", 40.0),
    ("a100", 40.0),  # ambiguous bare ``a100`` — treat as 40gb conservatively
    ("l40", 48.0),
    ("a10", 24.0),
    ("l4", 24.0),
    ("t4", 16.0),
)


def _gpu_memory_gb_for_profile(profile_name: str) -> float:
    """Match a profile name to a coarse GPU memory size, defaulting to
    :data:`_DEFAULT_GPU_MEMORY_GB` (L4) when no hint matches.
    """
    lower = profile_name.lower()
    for hint, gb in _GPU_MEMORY_GB_BY_PROFILE_HINT:
        if hint in lower:
            return gb
    return _DEFAULT_GPU_MEMORY_GB


def _coarse_kv_bytes_per_token_for(sie_id: str) -> int | None:
    """Return a coarse ``kv_bytes_per_token`` for a known model family.

    Returns ``None`` for unknown families — the over-subscription
    warning is skipped silently rather than guessing.
    """
    lower = sie_id.lower()
    if "qwen3-4b" in lower:
        return _KV_BYTES_PER_TOKEN_BY_FAMILY["qwen3-4b"]
    return None


def _maybe_warn_oversubscribed_budget(
    *,
    sie_id: str,
    profile_name: str,
    effective_budget: int,
    profile: "ProfileConfig",
    parent: "ProfileConfig | None",
) -> None:
    """Emit a ``UserWarning`` + structured log when ``kv_budget_tokens`` x
    a conservative concurrency factor exceeds the coarsely-derivable KV
    capacity for the model family. Warning, not error — operators can
    override (e.g. via larger GPUs or a tighter ``mem_fraction_static``).
    """
    kv_bytes_per_token = _coarse_kv_bytes_per_token_for(sie_id)
    if kv_bytes_per_token is None:
        # The coarse per-family table is intentionally narrow — the
        # calibration follow-up publishes calibrated values per family.
        # Surface the skip so
        # operators of unrecognised models can spot the gap and either
        # add a constant or open an issue.
        logger.debug(
            "kv_bytes_per_token unknown for %s; over-subscription guard skipped for profile '%s'",
            sie_id,
            profile_name,
        )
        return

    # Resolve effective loadtime (fieldwise: child non-empty wins).
    loadtime: dict[str, Any] = {}
    if parent is not None:
        loadtime = dict(parent.adapter_options.loadtime)
    if profile.adapter_options.loadtime:
        loadtime = dict(profile.adapter_options.loadtime)

    mem_fraction_static = loadtime.get("mem_fraction_static")
    if not isinstance(mem_fraction_static, int | float):
        return

    # Coarse: assume the documented in-flight estimate is 4 concurrent
    # generations (source-spec language). A higher concurrency makes
    # the budget more easily oversubscribed.
    in_flight_estimate = 4
    gpu_memory_gb = _gpu_memory_gb_for_profile(profile_name)
    derivable_bytes = float(mem_fraction_static) * gpu_memory_gb * 1024**3
    derivable_tokens = int(derivable_bytes / kv_bytes_per_token)
    needed_tokens = effective_budget * in_flight_estimate
    if needed_tokens > derivable_tokens:
        msg = (
            f"Profile '{profile_name}' on '{sie_id}': "
            f"kv_budget_tokens={effective_budget} * in_flight_estimate={in_flight_estimate} = "
            f"{needed_tokens} tokens exceeds the coarse derivable budget of "
            f"~{derivable_tokens} tokens (mem_fraction_static={mem_fraction_static}, "
            f"kv_bytes_per_token≈{kv_bytes_per_token}, "
            f"assumed_gpu_memory_gb={gpu_memory_gb}). "
            "Operators can override — this is a warning, not an error. "
            "Use measured workload and GPU data to calibrate this value."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(
            "kv_budget_tokens may be oversubscribed for %s/%s: budget=%d, in_flight_estimate=%d, derivable=%d",
            sie_id,
            profile_name,
            effective_budget,
            in_flight_estimate,
            derivable_tokens,
            extra={
                "model": sie_id,
                "profile": profile_name,
                "kv_budget_tokens": effective_budget,
                "derivable_tokens": derivable_tokens,
            },
        )


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
    _synthetic_profile_variant_source: tuple[str, str] | None = PrivateAttr(default=None)

    sie_id: str
    hf_id: str | None = None
    hf_revision: str | None = None
    hf_tokenizer_dependencies: dict[str, str] = Field(default_factory=dict)
    weights_path: Path | None = None
    package_backed: bool = False
    pool: str | None = None
    inputs: InputModalities = InputModalities()
    tasks: Tasks
    max_sequence_length: int | None = None
    profiles: dict[str, ProfileConfig]

    @property
    def synthetic_profile_variant_source(self) -> tuple[str, str] | None:
        return self._synthetic_profile_variant_source

    @model_validator(mode="after")
    def validate_pool_name(self) -> "ModelConfig":
        if self.pool is None:
            return self

        pool = self.pool.strip().lower()
        if not pool:
            self.pool = None
            return self

        if (
            len(pool) > _MAX_POOL_NAME_LEN
            or pool == "_default"
            or not all(c.isascii() and (c.isalnum() or c in "_-") for c in pool)
        ):
            msg = "pool must use only [A-Za-z0-9_-], be at most 128 chars, and not be '_default'"
            raise ValueError(msg)

        self.pool = pool
        return self

    @model_validator(mode="after")
    def validate_weight_source(self) -> "ModelConfig":
        if self.package_backed:
            if self.hf_id is not None or self.weights_path is not None or self.hf_revision is not None:
                msg = "'package_backed' models must not set 'hf_id', 'weights_path', or 'hf_revision'"
                raise ValueError(msg)
            return self
        if self.hf_id is None and self.weights_path is None:
            msg = "At least one of 'hf_id', 'weights_path', or 'package_backed' must be set"
            raise ValueError(msg)
        return self

    def lora_revisions(self) -> dict[str, str | None]:
        """Every LoRA id any profile declares -> its pinned revision (``None`` = unpinned).

        Merge policy across profiles (#2113): the LoRA identity is the bare id
        (it is the public adapter id clients switch on), so one id maps to one
        revision. A pin beats a bare mention — declaring the pin on one profile
        does not force touching every profile that names the adapter — but two
        *different* explicit SHAs for the same id are a genuine contradiction
        and raise ``ValueError``.
        """
        refs: dict[str, str | None] = {}
        for name, profile in self.profiles.items():
            entries: list[tuple[str, str | None]] = []
            lora_paths = profile.adapter_options.loadtime.get("lora_paths")
            if isinstance(lora_paths, Mapping):
                entries.extend(lora_entry_ref(value) for value in lora_paths.values() if value)
            elif isinstance(lora_paths, (list, tuple)):
                entries.extend((str(value), None) for value in lora_paths if value)
            lora_id = profile.adapter_options.runtime.get("lora_id")
            if lora_id:
                entries.append((str(lora_id), None))
            for ref_id, revision in entries:
                if ref_id not in refs:
                    refs[ref_id] = revision
                elif revision is not None:
                    if refs[ref_id] is None:
                        refs[ref_id] = revision
                    elif refs[ref_id] != revision:
                        msg = (
                            f"Model '{self.sie_id}' pins LoRA '{ref_id}' to two different revisions "
                            f"({refs[ref_id]} vs {revision}, latest seen in profile '{name}'). One id "
                            "maps to one adapter; pin a single SHA or serve the second revision under "
                            "a different id."
                        )
                        raise ValueError(msg)
        return refs

    def speculative_draft_revisions(self) -> dict[str, str | None]:
        """Enabled speculative draft repos mapped to their optional immutable pins."""
        refs: dict[str, str | None] = {}
        for name, profile in self.profiles.items():
            speculative = profile.adapter_options.loadtime.get("speculative")
            if not isinstance(speculative, Mapping) or not speculative.get("enabled"):
                continue
            draft_model = speculative.get("draft_model")
            if not isinstance(draft_model, str) or not draft_model:
                continue
            revision = speculative.get("draft_model_revision")
            if draft_model not in refs or refs[draft_model] is None:
                refs[draft_model] = revision
            elif revision is not None and refs[draft_model] != revision:
                msg = (
                    f"Model '{self.sie_id}' pins speculative draft '{draft_model}' to two different "
                    f"revisions ({refs[draft_model]} vs {revision}, latest seen in profile '{name}')"
                )
                raise ValueError(msg)
        return refs

    @model_validator(mode="after")
    def validate_lora_revision_consistency(self) -> "ModelConfig":
        # Surfaces the conflicting-pin error at register/config-load time rather
        # than first at model load. See :meth:`lora_revisions` for the policy.
        self.lora_revisions()
        return self

    @model_validator(mode="after")
    def validate_speculative_draft_revision_consistency(self) -> "ModelConfig":
        self.speculative_draft_revisions()
        return self

    @model_validator(mode="after")
    def validate_profiles(self) -> "ModelConfig":
        if not self.profiles:
            msg = "'profiles' must contain at least one profile"
            raise ValueError(msg)

        # ``grammar_profile`` must name a real profile (it becomes the
        # ``{sie_id}:{grammar_profile}`` variant the gateway routes grammar
        # requests to). Rejecting an unknown name here turns a silent
        # no-op (the gateway would skip the rewrite) into a load-time error.
        if self.tasks.generate is not None and self.tasks.generate.grammar_profile is not None:
            gp = self.tasks.generate.grammar_profile
            if gp not in self.profiles:
                msg = (
                    f"tasks.generate.grammar_profile '{gp}' is not a defined profile. Available: {list(self.profiles)}"
                )
                raise ValueError(msg)
            if gp == "default":
                msg = "tasks.generate.grammar_profile must not be 'default' (it names a non-default variant profile)"
                raise ValueError(msg)
            if self.profiles[gp].grammar_profile is not None:
                msg = f"tasks.generate.grammar_profile target '{gp}' must not declare another grammar_profile"
                raise ValueError(msg)

        for name, profile in self.profiles.items():
            if profile.grammar_profile is not None:
                grammar_profile = profile.grammar_profile
                if self.tasks.generate is None:
                    msg = f"Profile '{name}' sets grammar_profile on a model without a generation task"
                    raise ValueError(msg)
                if name == "default":
                    msg = "Profile 'default' must use tasks.generate.grammar_profile rather than a profile-scoped fallback"
                    raise ValueError(msg)
                if grammar_profile not in self.profiles:
                    msg = (
                        f"Profile '{name}' grammar_profile '{grammar_profile}' is not defined. "
                        f"Available: {list(self.profiles)}"
                    )
                    raise ValueError(msg)
                if grammar_profile in {"default", name}:
                    msg = (
                        f"Profile '{name}' grammar_profile must name a non-default sibling profile, "
                        f"got '{grammar_profile}'"
                    )
                    raise ValueError(msg)
                if self.profiles[grammar_profile].grammar_profile is not None:
                    msg = (
                        f"Profile '{name}' grammar_profile target '{grammar_profile}' "
                        "must not declare another grammar_profile"
                    )
                    raise ValueError(msg)
            if profile.chat_template_kwargs is not None and self.tasks.generate is None:
                msg = f"Profile '{name}' sets 'chat_template_kwargs' on a model without a generation task"
                raise ValueError(msg)
            if profile.max_output_tokens is not None and self.tasks.generate is None:
                msg = f"Profile '{name}' sets 'max_output_tokens' on a model without a generation task"
                raise ValueError(msg)
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

        if self.tasks.generate is not None and self.tasks.generate.grammar_profile is not None:
            grammar_profile = self.tasks.generate.grammar_profile
            target = self._resolve_profile_uncached(grammar_profile)
            target_speculative = target.loadtime.get("speculative")
            if target_speculative is not None and (
                not isinstance(target_speculative, dict) or target_speculative.get("enabled") is not False
            ):
                msg = f"tasks.generate.grammar_profile target '{grammar_profile}' must not enable speculation"
                raise ValueError(msg)

        for name, profile in self.profiles.items():
            if profile.grammar_profile is None:
                continue
            source = self._resolve_profile_uncached(name)
            target = self._resolve_profile_uncached(profile.grammar_profile)
            source_loadtime = dict(source.loadtime)
            target_loadtime = dict(target.loadtime)
            source_loadtime.pop("speculative", None)
            target_loadtime.pop("speculative", None)
            target_speculative = target.loadtime.get("speculative")
            compatible = (
                source.adapter_path == target.adapter_path
                and source.max_batch_tokens == target.max_batch_tokens
                and source.compute_precision == target.compute_precision
                and source.kv_budget_tokens == target.kv_budget_tokens
                and source.max_output_tokens == target.max_output_tokens
                and source.chat_template_kwargs == target.chat_template_kwargs
                and target_speculative == {"enabled": False}
                and source_loadtime == target_loadtime
            )
            if not compatible:
                msg = (
                    f"Profile '{name}' grammar_profile '{profile.grammar_profile}' must preserve "
                    "adapter, precision, batch, KV/output limits, tokenizer mode, and load-time "
                    "settings while explicitly disabling speculation"
                )
                raise ValueError(msg)

        # KV-budget admission control. For models declaring
        # ``tasks.generate``, every profile (after parent merge) must
        # provide a positive ``kv_budget_tokens``. The actual
        # calibrated value lands in the calibration follow-up; until then operators may
        # carry a placeholder in YAML but missing/non-positive values
        # are a hard error pointing at the calibration deliverable.
        if self.tasks.generate is not None:
            for name, profile in self.profiles.items():
                effective_budget: int | None
                if profile.extends is not None:
                    parent = self.profiles[profile.extends]
                    effective_budget = (
                        profile.kv_budget_tokens if profile.kv_budget_tokens is not None else parent.kv_budget_tokens
                    )
                else:
                    effective_budget = profile.kv_budget_tokens
                if effective_budget is None:
                    msg = (
                        f"Profile '{name}' on a generation model "
                        f"('{self.sie_id}') is missing 'kv_budget_tokens'. "
                        "This is the per-worker KV-cache admission budget."
                    )
                    raise ValueError(msg)
                if not isinstance(effective_budget, int) or effective_budget <= 0:
                    msg = (
                        f"Profile '{name}' on a generation model "
                        f"('{self.sie_id}'): 'kv_budget_tokens' must be a "
                        f"positive int, got {effective_budget!r}."
                    )
                    raise ValueError(msg)

                # Coarse over-subscription warning. Derivation uses
                # ``loadtime.mem_fraction_static`` × an assumed GPU
                # capacity in GB, multiplied by a coarse
                # ``kv_bytes_per_token`` constant for known model
                # families. Operators can override; this is a
                # warning, not an error.
                _maybe_warn_oversubscribed_budget(
                    sie_id=self.sie_id,
                    profile_name=name,
                    effective_budget=effective_budget,
                    profile=profile,
                    parent=self.profiles[profile.extends] if profile.extends is not None else None,
                )

                parent = self.profiles[profile.extends] if profile.extends is not None else None
                effective_output_cap = (
                    profile.max_output_tokens
                    if profile.max_output_tokens is not None
                    else parent.max_output_tokens
                    if parent is not None
                    else None
                )
                if effective_output_cap is not None:
                    effective_loadtime = profile.adapter_options.loadtime or (
                        parent.adapter_options.loadtime if parent is not None else {}
                    )
                    effective_context = effective_loadtime.get(
                        "max_seq_length",
                        self.tasks.generate.context_length,
                    )
                    if effective_output_cap > effective_context:
                        msg = (
                            f"Profile '{name}' on generation model '{self.sie_id}' sets "
                            f"max_output_tokens={effective_output_cap}, exceeding its "
                            f"context_length={effective_context}"
                        )
                        raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_package_artifacts(self) -> "ModelConfig":
        declarations: set[PackageArtifactDeclaration] = set()
        for name in self.profiles:
            loadtime = self._effective_package_artifact_loadtime(name)
            if not self.package_backed and has_package_artifact_declaration(loadtime):
                raise ValueError("package artifact declarations require 'package_backed: true'")
            declarations.add(parse_package_artifact_declaration(loadtime))
        if self.package_backed and len(declarations) != 1:
            raise ValueError("all profiles of a package-backed model must declare identical package artifacts")
        return self

    @model_validator(mode="after")
    def validate_serving_artifacts(self) -> "ModelConfig":
        resolved_profiles = {name: self._resolve_profile_uncached(name) for name in self.profiles}
        declarations = {
            name: parse_serving_artifact_declaration(dict(resolved.loadtime))
            for name, resolved in resolved_profiles.items()
        }
        if not any(declarations.values()):
            return self
        if self.package_backed or self.weights_path is not None:
            raise ValueError("derived serving artifacts require an HF source checkpoint")
        if self.hf_id is None or not is_immutable_revision(self.hf_revision):
            raise ValueError("derived serving artifacts require source hf_id and an immutable 40-char hf_revision")
        for name, declaration in declarations.items():
            if declaration is None:
                continue
            # Bare CT2 ``int8`` leaves the floating-point accumulation type to
            # runtime/device selection. Only the explicit compute types can be
            # compared to SIE's declared profile precision without guessing.
            expected_precision = _SERVING_ARTIFACT_PRECISION.get(declaration.compute_type)
            if expected_precision is None:
                continue
            actual_precision = resolved_profiles[name].compute_precision
            if actual_precision is not None and actual_precision != expected_precision:
                raise ValueError(
                    f"Profile '{name}' compute_precision={actual_precision!r} does not match "
                    f"serving artifact compute_type={declaration.compute_type!r}; "
                    f"expected {expected_precision!r}"
                )
        return self

    def serving_artifact_declaration(self, profile: str = "default") -> ServingArtifactDeclaration | None:
        """Return the effective profile's immutable derived-artifact declaration."""
        loadtime = dict(self.resolve_profile(profile).loadtime)
        return parse_serving_artifact_declaration(loadtime)

    def _effective_package_artifact_loadtime(self, name: str) -> dict[str, Any]:
        # Keep artifact validation on the exact same inheritance/replacement
        # path as adapter construction.  Do not duplicate profile resolution
        # semantics here: that would let future resolution changes silently
        # validate a different declaration than the worker consumes.
        return dict(self._resolve_profile_uncached(name).loadtime)

    @property
    def package_artifact_declaration(self) -> PackageArtifactDeclaration:
        first_profile = next(iter(self.profiles))
        return parse_package_artifact_declaration(self._effective_package_artifact_loadtime(first_profile))

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
                kv_budget_tokens=profile.kv_budget_tokens,
                admission_enabled=profile.admission_enabled,
                max_output_tokens=profile.max_output_tokens,
                chat_template_kwargs=(
                    MappingProxyType(dict(profile.chat_template_kwargs))
                    if profile.chat_template_kwargs is not None
                    else None
                ),
                grammar_profile=profile.grammar_profile,
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

        # Child non-None overrides parent for the admission fields.
        kv_budget_tokens = profile.kv_budget_tokens if profile.kv_budget_tokens is not None else parent.kv_budget_tokens
        admission_enabled = (
            profile.admission_enabled if profile.admission_enabled is not None else parent.admission_enabled
        )
        max_output_tokens = (
            profile.max_output_tokens if profile.max_output_tokens is not None else parent.max_output_tokens
        )
        chat_template_kwargs = (
            profile.chat_template_kwargs if profile.chat_template_kwargs is not None else parent.chat_template_kwargs
        )
        # Routing metadata is profile-local rather than inherited. An alias
        # must opt into the same fallback explicitly, and the fallback profile
        # itself must not inherit a route back to itself.
        grammar_profile = profile.grammar_profile

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
            kv_budget_tokens=kv_budget_tokens,
            admission_enabled=admission_enabled,
            max_output_tokens=max_output_tokens,
            chat_template_kwargs=(
                MappingProxyType(dict(chat_template_kwargs)) if chat_template_kwargs is not None else None
            ),
            grammar_profile=grammar_profile,
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
        if self.tasks.generate is not None:
            result.append("tokens")
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
