from pathlib import Path

import pytest
from pydantic import ValidationError
from sie_server.config.engine import EngineConfig
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    ExtractTask,
    GenerateCapabilities,
    GenerateTask,
    InputModalities,
    ModelConfig,
    ProfileConfig,
    ResolvedProfile,
    ScoreTask,
    Tasks,
)
from sie_server.config.package_artifacts import PackageArtifactMode


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_defaults(self) -> None:
        """EngineConfig has sensible defaults."""
        config = EngineConfig()
        # Note: max_batch_tokens is per-model (in ModelConfig), not engine-level
        assert config.max_batch_requests == 64
        assert config.max_batch_wait_ms == 15
        assert config.coalesce_ms == 15.0
        assert config.coalesce_ratio == 0.5
        assert config.max_concurrent_requests == 512
        assert config.memory_pressure_threshold_percent == 95
        assert config.max_loras_per_model == 10
        assert config.preprocessor_workers == 4
        assert config.attention_backend == "auto"
        assert config.default_compute_precision == "float16"
        assert config.instrumentation is False
        assert config.models_dir == Path("./models")
        assert config.adaptive_batching.min_wait_ms == 15.0
        assert config.adaptive_batching.max_wait_ms == 50.0
        assert config.adaptive_batching.starvation_recovery_enabled is True
        assert config.adaptive_batching.starvation_window == 20
        assert config.adaptive_batching.starvation_batch_size == 1

    def test_custom_values(self) -> None:
        """EngineConfig accepts custom values."""
        config = EngineConfig(
            max_batch_requests=128,
            attention_backend="flash_attention_2",
            default_compute_precision="bfloat16",
        )
        assert config.max_batch_requests == 128
        assert config.attention_backend == "flash_attention_2"
        assert config.default_compute_precision == "bfloat16"

    def test_max_batch_requests_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_MAX_BATCH_REQUESTS overrides the batch-depth cap (the sglang-embed deep-batch lane knob).

        The managed embed lane bakes this env into the container image so the
        engine's batcher packs deeper forwards; proves the env-wiring the deploy
        relies on (default 64 -> 128).
        """
        assert EngineConfig().max_batch_requests == 64
        monkeypatch.setenv("SIE_MAX_BATCH_REQUESTS", "128")
        assert EngineConfig().max_batch_requests == 128

    def test_invalid_attention_backend(self) -> None:
        """Invalid attention backend is rejected."""
        with pytest.raises(ValidationError):
            EngineConfig(attention_backend="invalid")  # type: ignore

    def test_invalid_precision(self) -> None:
        """Invalid compute precision is rejected."""
        with pytest.raises(ValidationError):
            EngineConfig(default_compute_precision="fp16")  # type: ignore

    def test_memory_threshold_bounds(self) -> None:
        """Memory threshold must be 50-99%."""
        with pytest.raises(ValidationError):
            EngineConfig(memory_pressure_threshold_percent=100)

        with pytest.raises(ValidationError):
            EngineConfig(memory_pressure_threshold_percent=49)


class TestOomRecoveryConfig:
    """OOM-recovery sub-config and the SIE_DISABLE_OOM_RECOVERY kill switch."""

    def test_defaults(self) -> None:
        """Recovery is on, strategy ordered cheap→disruptive, depth=4."""
        config = EngineConfig()
        assert config.oom_recovery.enabled is True
        assert config.oom_recovery.strategy == ["cache_clear", "evict_lru", "split_batch"]
        assert config.oom_recovery.max_split_depth == 4
        assert config.oom_recovery.retry_after_s == 5

    def test_to_runtime_preserves_fields(self) -> None:
        """``to_runtime`` produces a dataclass identical to the pydantic view."""
        config = EngineConfig()
        runtime = config.oom_recovery.to_runtime()
        assert runtime.enabled is True
        assert runtime.max_split_depth == 4
        assert runtime.retry_after_s == 5
        # Strategy is a tuple of OomRecoveryAction enum values.
        assert tuple(a.value for a in runtime.strategy) == (
            "cache_clear",
            "evict_lru",
            "split_batch",
        )

    @pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes", "YES"])
    def test_kill_switch_disables_recovery(self, flag_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``SIE_DISABLE_OOM_RECOVERY`` overrides the default-enabled state."""
        monkeypatch.setenv("SIE_DISABLE_OOM_RECOVERY", flag_value)
        # Avoid the env-file loading interfering with this test.
        config = EngineConfig()
        assert config.oom_recovery.enabled is False

    @pytest.mark.parametrize("flag_value", ["", "0", "false", "no", "off", "anything"])
    def test_kill_switch_unset_or_falsy_keeps_default(self, flag_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only ``1``/``true``/``yes`` flip the switch; other values leave default."""
        monkeypatch.setenv("SIE_DISABLE_OOM_RECOVERY", flag_value)
        config = EngineConfig()
        assert config.oom_recovery.enabled is True

    def test_kill_switch_wins_over_explicit_enabled_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Top-level kill switch wins over an explicit nested ``enabled=true``.

        Operators reaching for the convenience flag during an incident
        expect it to take effect even if the nested setting was previously
        set explicitly. The validator runs *after* nested settings parse
        and intentionally overrides them.
        """
        monkeypatch.setenv("SIE_DISABLE_OOM_RECOVERY", "1")
        monkeypatch.setenv("SIE_OOM_RECOVERY__ENABLED", "true")
        config = EngineConfig()
        assert config.oom_recovery.enabled is False

    def test_nested_env_var_alone_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the kill switch, the nested env var is honoured."""
        monkeypatch.delenv("SIE_DISABLE_OOM_RECOVERY", raising=False)
        monkeypatch.setenv("SIE_OOM_RECOVERY__ENABLED", "false")
        config = EngineConfig()
        assert config.oom_recovery.enabled is False

    def test_max_split_depth_validation_bounds(self) -> None:
        """``max_split_depth`` must be 0-8."""
        with pytest.raises(ValidationError):
            EngineConfig.model_validate({"oom_recovery": {"max_split_depth": -1}})
        with pytest.raises(ValidationError):
            EngineConfig.model_validate({"oom_recovery": {"max_split_depth": 9}})

    def test_retry_after_s_validation_bounds(self) -> None:
        """``retry_after_s`` must be 1-60 seconds."""
        with pytest.raises(ValidationError):
            EngineConfig.model_validate({"oom_recovery": {"retry_after_s": 0}})
        with pytest.raises(ValidationError):
            EngineConfig.model_validate({"oom_recovery": {"retry_after_s": 61}})


def _make_config(
    sie_id: str = "test-model",
    *,
    hf_id: str | None = "org/model",
    weights_path: Path | None = None,
    dense_dim: int | None = 768,
    sparse_dim: int | None = None,
    multivector_dim: int | None = None,
    score: bool = False,
    extract: bool = False,
    adapter_path: str = "sie_server.adapters.test:TestAdapter",
    max_batch_tokens: int = 8192,
    max_sequence_length: int | None = None,
    compute_precision: str | None = None,
    profiles: dict[str, ProfileConfig] | None = None,
    pool: str | None = None,
) -> ModelConfig:
    encode = None
    if any(dim is not None for dim in (dense_dim, sparse_dim, multivector_dim)):
        encode = EncodeTask(
            dense=EmbeddingDim(dim=dense_dim) if dense_dim is not None else None,
            sparse=EmbeddingDim(dim=sparse_dim) if sparse_dim is not None else None,
            multivector=EmbeddingDim(dim=multivector_dim) if multivector_dim is not None else None,
        )
    tasks = Tasks(
        encode=encode,
        score=ScoreTask() if score else None,
        extract=ExtractTask() if extract else None,
    )
    if profiles is None:
        profiles = {
            "default": ProfileConfig(
                adapter_path=adapter_path,
                max_batch_tokens=max_batch_tokens,
                compute_precision=compute_precision,  # type: ignore
            ),
        }
    return ModelConfig(
        sie_id=sie_id,
        hf_id=hf_id,
        weights_path=weights_path,
        tasks=tasks,
        pool=pool,
        max_sequence_length=max_sequence_length,
        profiles=profiles,
    )


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_minimal_config(self) -> None:
        """ModelConfig with minimal required fields."""
        config = _make_config()
        assert config.sie_id == "test-model"
        assert config.hf_id == "org/model"
        assert config.tasks.encode.dense.dim == 768  # type: ignore
        assert config.hf_tokenizer_dependencies == {}

    def test_hf_tokenizer_dependencies_accepts_mapping(self) -> None:
        """ModelConfig accepts a tokenizer repository-to-revision mapping."""
        dependencies = {
            "microsoft/mdeberta-v3-base": "a0484667b22365f84929a935b5e50a51f71f159d",
        }
        payload = _make_config().model_dump(mode="json")
        payload["hf_tokenizer_dependencies"] = dependencies

        config = ModelConfig.model_validate(payload)

        assert config.hf_tokenizer_dependencies == dependencies

    def test_hf_tokenizer_dependencies_round_trip(self) -> None:
        """Tokenizer dependency pins survive JSON serialization and validation."""
        dependencies = {
            "microsoft/deberta-v3-large": "64a8c8eab3e352a784c658aef62be1662607476f",
        }
        payload = _make_config().model_dump(mode="json")
        payload["hf_tokenizer_dependencies"] = dependencies
        config = ModelConfig.model_validate(payload)

        serialized = config.model_dump(mode="json")

        assert serialized["hf_tokenizer_dependencies"] == dependencies
        assert ModelConfig.model_validate(serialized).hf_tokenizer_dependencies == dependencies

    @pytest.mark.parametrize(
        "invalid_dependencies",
        [
            [],
            {"microsoft/deberta-v3-large": 123},
        ],
    )
    def test_hf_tokenizer_dependencies_rejects_invalid_types(self, invalid_dependencies: object) -> None:
        """Tokenizer dependencies fail closed when the mapping shape is invalid."""
        payload = _make_config().model_dump(mode="json")
        payload["hf_tokenizer_dependencies"] = invalid_dependencies

        with pytest.raises(ValidationError, match="hf_tokenizer_dependencies"):
            ModelConfig.model_validate(payload)

    def test_local_weights(self) -> None:
        """ModelConfig can use local weights."""
        config = _make_config(
            "local-model",
            hf_id=None,
            weights_path=Path("/data/models/test"),
        )
        assert config.weights_path == Path("/data/models/test")
        assert config.hf_id is None

    def test_missing_weight_source_rejected(self) -> None:
        """Model without weight source is rejected."""
        with pytest.raises(ValidationError, match=r"hf_id.*weights_path"):
            ModelConfig(
                sie_id="no-weights",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={"default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192)},
            )

    def test_package_backed_allows_no_weights(self) -> None:
        """package_backed=True lets adapters that ship their own weights skip hf_id/weights_path."""
        config = ModelConfig(
            sie_id="docling",
            package_backed=True,
            tasks=Tasks(extract=ExtractTask()),
            inputs=InputModalities(text=False, document=True),
            profiles={"default": ProfileConfig(adapter_path="sie_server.adapters.docling:Cls", max_batch_tokens=1)},
        )
        assert config.package_backed is True
        assert config.hf_id is None
        assert config.weights_path is None

    def test_package_backed_rejects_hf_id(self) -> None:
        """package_backed and hf_id are mutually exclusive — adapter ships weights itself."""
        with pytest.raises(ValidationError, match=r"package_backed"):
            ModelConfig(
                sie_id="bad",
                package_backed=True,
                hf_id="org/model",
                tasks=Tasks(extract=ExtractTask()),
                profiles={"default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=1)},
            )

    def test_package_backed_rejects_weights_path(self) -> None:
        with pytest.raises(ValidationError, match=r"package_backed"):
            ModelConfig(
                sie_id="bad",
                package_backed=True,
                weights_path=Path("/data/x"),
                tasks=Tasks(extract=ExtractTask()),
                profiles={"default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=1)},
            )

    def test_package_backed_rejects_hf_revision(self) -> None:
        """hf_revision pins a HF snapshot — meaningless for package-backed adapters."""
        with pytest.raises(ValidationError, match=r"hf_revision"):
            ModelConfig(
                sie_id="bad",
                package_backed=True,
                hf_revision="abc123",
                tasks=Tasks(extract=ExtractTask()),
                profiles={"default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=1)},
            )

    def test_package_backed_declares_live_external_artifacts(self) -> None:
        config = ModelConfig(
            sie_id="package/model",
            package_backed=True,
            tasks=Tasks(extract=ExtractTask()),
            profiles={
                "default": ProfileConfig(
                    adapter_path="mod:Cls",
                    max_batch_tokens=1,
                    adapter_options=AdapterOptions(loadtime={"package_artifact_mode": "live"}),
                )
            },
        )

        assert config.package_artifact_declaration.mode == PackageArtifactMode.LIVE

    def test_staged_package_artifacts_require_complete_manifest_identity(self) -> None:
        with pytest.raises(ValidationError, match="64-hex manifest sha256"):
            ModelConfig(
                sie_id="package/model",
                package_backed=True,
                tasks=Tasks(extract=ExtractTask()),
                profiles={
                    "default": ProfileConfig(
                        adapter_path="mod:Cls",
                        max_batch_tokens=1,
                        adapter_options=AdapterOptions(
                            loadtime={
                                "package_artifact_mode": "staged",
                                "package_artifact_manifest_path": "/models/package/model/manifest.json",
                            }
                        ),
                    )
                },
            )

    def test_package_artifact_declaration_must_match_across_profiles(self) -> None:
        with pytest.raises(ValidationError, match="identical package artifacts"):
            ModelConfig(
                sie_id="package/model",
                package_backed=True,
                tasks=Tasks(extract=ExtractTask()),
                profiles={
                    "default": ProfileConfig(
                        adapter_path="mod:Cls",
                        max_batch_tokens=1,
                        adapter_options=AdapterOptions(loadtime={"package_artifact_mode": "live"}),
                    ),
                    "bundled": ProfileConfig(
                        adapter_path="mod:Cls",
                        max_batch_tokens=1,
                    ),
                },
            )

    def test_package_artifact_declaration_uses_resolved_profile_loadtime(self) -> None:
        config = ModelConfig(
            sie_id="package/model",
            package_backed=True,
            tasks=Tasks(extract=ExtractTask()),
            profiles={
                "default": ProfileConfig(
                    adapter_path="mod:Cls",
                    max_batch_tokens=1,
                    adapter_options=AdapterOptions(loadtime={"package_artifact_mode": "live"}),
                ),
                "inherited": ProfileConfig(extends="default"),
            },
        )

        assert config.resolve_profile("inherited").loadtime["package_artifact_mode"] == "live"
        assert config.package_artifact_declaration.mode == PackageArtifactMode.LIVE

    def test_child_loadtime_replacement_must_repeat_package_artifact_declaration(self) -> None:
        with pytest.raises(ValidationError, match="identical package artifacts"):
            ModelConfig(
                sie_id="package/model",
                package_backed=True,
                tasks=Tasks(extract=ExtractTask()),
                profiles={
                    "default": ProfileConfig(
                        adapter_path="mod:Cls",
                        max_batch_tokens=1,
                        adapter_options=AdapterOptions(loadtime={"package_artifact_mode": "live"}),
                    ),
                    "replacement": ProfileConfig(
                        extends="default",
                        adapter_options=AdapterOptions(loadtime={"unrelated_option": True}),
                    ),
                },
            )

    def test_non_package_model_rejects_package_artifact_declaration(self) -> None:
        with pytest.raises(ValidationError, match="require 'package_backed: true'"):
            ModelConfig(
                sie_id="org/model",
                hf_id="org/model",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={
                    "default": ProfileConfig(
                        adapter_path="mod:Cls",
                        max_batch_tokens=1,
                        adapter_options=AdapterOptions(loadtime={"package_artifact_mode": "live"}),
                    )
                },
            )

    def test_empty_profiles_rejected(self) -> None:
        """Model config must define at least one profile."""
        with pytest.raises(ValidationError, match=r"profiles"):
            ModelConfig(
                sie_id="no-profiles",
                hf_id="org/model",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={},
            )

    def test_default_profile_needs_adapter_path(self) -> None:
        """Default profile must have adapter_path."""
        with pytest.raises(ValidationError, match=r"adapter_path"):
            ModelConfig(
                sie_id="no-adapter",
                hf_id="org/model",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={"default": ProfileConfig(max_batch_tokens=8192)},
            )

    def test_default_profile_needs_max_batch_tokens(self) -> None:
        """Default profile must have max_batch_tokens."""
        with pytest.raises(ValidationError, match=r"max_batch_tokens"):
            ModelConfig(
                sie_id="no-batch",
                hf_id="org/model",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={"default": ProfileConfig(adapter_path="mod:Cls")},
            )

    def test_full_config(self) -> None:
        """ModelConfig with all fields."""
        config = _make_config(
            "bge-m3",
            hf_id="BAAI/bge-m3",
            dense_dim=1024,
            sparse_dim=250002,
            multivector_dim=1024,
            max_sequence_length=8192,
            adapter_path="sie_server.adapters.bge_m3:BGEM3Adapter",
            compute_precision="float16",
        )
        # Backward-compat properties
        assert config.outputs == ["dense", "sparse", "multivector"]
        assert config.dims["dense"] == 1024
        assert config.dims["sparse"] == 250002
        # Direct new-schema access
        assert config.tasks.encode.dense.dim == 1024  # type: ignore
        assert config.tasks.encode.sparse.dim == 250002  # type: ignore
        assert config.max_sequence_length == 8192

    def test_extra_fields_rejected(self) -> None:
        """ModelConfig rejects unknown fields."""
        with pytest.raises(ValidationError):
            ModelConfig(
                sie_id="test",
                hf_id="org/model",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={"default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192)},
                unknown_field="value",  # type: ignore
            )

    def test_pool_field_normalized(self) -> None:
        """ModelConfig accepts pool assignment and canonicalizes casing."""
        config = _make_config(pool="Customer-A")
        assert config.pool == "customer-a"

    def test_pool_field_rejects_invalid_subject_token(self) -> None:
        """Model pool names must be safe for routing labels and subjects."""
        with pytest.raises(ValidationError, match="pool must use only"):
            _make_config(pool="customer.a")

    def test_inputs_default(self) -> None:
        """Default inputs is text-only."""
        config = _make_config()
        assert config.inputs.text is True
        assert config.inputs.image is False

    def test_inputs_multimodal(self) -> None:
        """InputModalities can include image."""
        config = ModelConfig(
            sie_id="clip",
            hf_id="openai/clip",
            inputs=InputModalities(text=True, image=True),
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=512))),
            profiles={"default": ProfileConfig(adapter_path="mod:Clip", max_batch_tokens=4096)},
        )
        assert config.inputs.text is True
        assert config.inputs.image is True

    def test_inputs_document_modality(self) -> None:
        """InputModalities can advertise document-parser inputs (PDF/DOCX/HTML)."""
        config = ModelConfig(
            sie_id="docling",
            hf_id="docling-project/docling",
            inputs=InputModalities(text=False, document=True),
            tasks=Tasks(extract=ExtractTask()),
            profiles={"default": ProfileConfig(adapter_path="mod:Docling", max_batch_tokens=1)},
        )
        assert config.inputs.document is True
        assert config.inputs.to_list() == ["document"]

    def test_backward_compat_name(self) -> None:
        """Name property returns sie_id."""
        config = _make_config("my-model")
        assert config.name == "my-model"

    def test_backward_compat_outputs(self) -> None:
        """Outputs property derives from tasks."""
        config = _make_config(dense_dim=768, sparse_dim=30000, score=True, extract=True)
        assert "dense" in config.outputs
        assert "sparse" in config.outputs
        assert "score" in config.outputs
        assert "json" in config.outputs

    def test_backward_compat_dims(self) -> None:
        """Dims property returns dict of dimensions."""
        config = _make_config(dense_dim=768, sparse_dim=30000, multivector_dim=128)
        assert config.dims == {"dense": 768, "sparse": 30000, "multivector": 128}

    def test_score_task(self) -> None:
        """ModelConfig with score task."""
        config = _make_config(dense_dim=None, score=True)
        assert config.tasks.score is not None
        assert "score" in config.outputs

    def test_extract_task(self) -> None:
        """ModelConfig with extract task."""
        config = _make_config(dense_dim=None, extract=True)
        assert config.tasks.extract is not None
        assert "json" in config.outputs

    def test_generate_task_accepted(self) -> None:
        """ModelConfig with the walking-skeleton generate task validates and exposes 'tokens' output."""
        config = ModelConfig(
            sie_id="Qwen/Qwen3-4B-Instruct",
            hf_id="Qwen/Qwen3-4B-Instruct",
            tasks=Tasks(
                generate=GenerateTask(
                    context_length=32768,
                    max_output_tokens=4096,
                    capabilities=GenerateCapabilities(grammar=[], streaming=True, tools=False),
                ),
            ),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                    max_batch_tokens=16384,
                    kv_budget_tokens=8192,
                ),
            },
        )
        assert config.tasks.generate is not None
        assert config.tasks.generate.context_length == 32768
        assert config.tasks.generate.max_output_tokens == 4096
        assert "tokens" in config.outputs

    def test_generate_task_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            GenerateTask(
                context_length=32768,
                max_output_tokens=4096,
                # ``unknown`` is not declared; extra='forbid' rejects it.
                unknown="x",  # type: ignore
            )

    def test_generate_capabilities_accepts_ebnf(self) -> None:
        """``ebnf`` was added to the accepted grammar list when
        Outlines / XGrammar EBNF support landed. Prior to that this test
        asserted rejection; it's flipped to acceptance as the regression
        guard so a future refactor doesn't silently drop EBNF support.
        """
        caps = GenerateCapabilities(grammar=["ebnf"])
        assert "ebnf" in caps.grammar

    def test_generate_capabilities_rejects_unknown_grammar(self) -> None:
        with pytest.raises(ValidationError):
            GenerateCapabilities(grammar=["totally-not-a-grammar"])  # type: ignore


class TestEngineConfigLoRA:
    """Tests for LoRA configuration in EngineConfig."""

    def test_max_loras_per_model_default(self) -> None:
        """Default max_loras_per_model is 10."""
        config = EngineConfig()
        assert config.max_loras_per_model == 10

    def test_max_loras_per_model_custom(self) -> None:
        """Custom max_loras_per_model is accepted."""
        config = EngineConfig(max_loras_per_model=20)
        assert config.max_loras_per_model == 20

    def test_max_loras_per_model_minimum(self) -> None:
        """max_loras_per_model must be at least 1."""
        with pytest.raises(ValidationError):
            EngineConfig(max_loras_per_model=0)


class TestProfileConfig:
    """Tests for ProfileConfig (new schema)."""

    def test_default_profile(self) -> None:
        """ProfileConfig with adapter_path and max_batch_tokens."""
        profile = ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192)
        assert profile.adapter_path == "mod:Cls"
        assert profile.max_batch_tokens == 8192

    def test_extends(self) -> None:
        """ProfileConfig can extend another profile."""
        profile = ProfileConfig(extends="default", max_batch_tokens=4096)
        assert profile.extends == "default"

    def test_adapter_options(self) -> None:
        """ProfileConfig can have adapter options."""
        profile = ProfileConfig(
            adapter_path="mod:Cls",
            max_batch_tokens=8192,
            adapter_options=AdapterOptions(
                loadtime={"trust_remote_code": True},
                runtime={"instruction": "Retrieve relevant docs"},
            ),
        )
        assert profile.adapter_options.loadtime == {"trust_remote_code": True}
        assert profile.adapter_options.runtime == {"instruction": "Retrieve relevant docs"}

    def test_compute_precision(self) -> None:
        """ProfileConfig can override compute precision."""
        profile = ProfileConfig(
            adapter_path="mod:Cls",
            max_batch_tokens=8192,
            compute_precision="bfloat16",
        )
        assert profile.compute_precision == "bfloat16"

    def test_extra_fields_rejected(self) -> None:
        """ProfileConfig rejects unknown fields."""
        with pytest.raises(ValidationError):
            ProfileConfig(
                adapter_path="mod:Cls",
                max_batch_tokens=8192,
                unknown="value",  # type: ignore
            )


class TestModelConfigProfiles:
    """Tests for profiles in ModelConfig."""

    def test_model_with_profiles(self) -> None:
        """ModelConfig can define multiple profiles."""
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/model",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192),
                "fast": ProfileConfig(extends="default", max_batch_tokens=4096),
            },
        )
        assert "default" in config.profiles
        assert "fast" in config.profiles
        assert config.profiles["fast"].extends == "default"

    def test_model_can_be_profile_only_without_default(self) -> None:
        """Profile-only configs are valid and expose explicit variants only."""
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/model",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "candle": ProfileConfig(
                    adapter_path="sie_server_rust.adapters.candle:CandleEmbeddingAdapter",
                    max_batch_tokens=8192,
                ),
            },
        )

        assert set(config.profiles) == {"candle"}
        assert config.resolve_profile("candle").adapter_path == "sie_server_rust.adapters.candle:CandleEmbeddingAdapter"

    def test_resolve_default_profile(self) -> None:
        """resolve_profile returns ResolvedProfile for default."""
        config = _make_config(adapter_path="mod:Cls", max_batch_tokens=8192)
        resolved = config.resolve_profile("default")
        assert isinstance(resolved, ResolvedProfile)
        assert resolved.adapter_path == "mod:Cls"
        assert resolved.max_batch_tokens == 8192

    def test_resolve_child_profile_inherits(self) -> None:
        """Child profile inherits from parent."""
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/model",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192),
                "fast": ProfileConfig(extends="default", max_batch_tokens=4096),
            },
        )
        resolved = config.resolve_profile("fast")
        assert resolved.adapter_path == "mod:Cls"  # inherited
        assert resolved.max_batch_tokens == 4096  # overridden

    def test_resolve_child_profile_overrides_adapter_options(self) -> None:
        """Child profile replaces adapter_options when non-empty."""
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/model",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="mod:Cls",
                    max_batch_tokens=8192,
                    adapter_options=AdapterOptions(runtime={"instruction": "parent"}),
                ),
                "child": ProfileConfig(
                    extends="default",
                    adapter_options=AdapterOptions(runtime={"instruction": "child"}),
                ),
            },
        )
        resolved = config.resolve_profile("child")
        assert resolved.runtime == {"instruction": "child"}

    def test_resolve_child_profile_inherits_empty_runtime(self) -> None:
        """An explicit empty runtime preserves the parent's runtime."""
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/model",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="mod:Cls",
                    max_batch_tokens=8192,
                    adapter_options=AdapterOptions(runtime={"output_similarity": {"dense": "dot"}}),
                ),
                "child": ProfileConfig(
                    extends="default",
                    adapter_options=AdapterOptions(runtime={}),
                ),
            },
        )

        assert config.resolve_profile("child").runtime == {"output_similarity": {"dense": "dot"}}

    def test_resolve_missing_profile_raises(self) -> None:
        """resolve_profile raises for unknown profile."""
        config = _make_config()
        with pytest.raises(ValueError, match="not found"):
            config.resolve_profile("nonexistent")

    def test_chaining_not_allowed(self) -> None:
        """Profile chaining (extends on extends) is rejected at construction time."""
        with pytest.raises(ValidationError, match="chaining"):
            ModelConfig(
                sie_id="test-model",
                hf_id="org/model",
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={
                    "default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192),
                    "mid": ProfileConfig(extends="default", max_batch_tokens=4096),
                    "deep": ProfileConfig(extends="mid"),
                },
            )


class TestKvBudgetTokensValidator:
    """Validator for ``kv_budget_tokens`` on generation profiles."""

    @staticmethod
    def _make_gen_config(profile: ProfileConfig, *, extra: dict[str, ProfileConfig] | None = None) -> ModelConfig:
        profiles = {"default": profile}
        if extra:
            profiles.update(extra)
        return ModelConfig(
            sie_id="Qwen/Qwen3-4B-Instruct-2507",
            hf_id="Qwen/Qwen3-4B-Instruct-2507",
            tasks=Tasks(
                generate=GenerateTask(
                    context_length=32768,
                    max_output_tokens=4096,
                    capabilities=GenerateCapabilities(grammar=[], streaming=True, tools=False),
                ),
            ),
            profiles=profiles,
        )

    def test_missing_kv_budget_on_gen_profile_rejected(self) -> None:
        with pytest.raises(ValidationError, match="kv_budget_tokens"):
            self._make_gen_config(
                ProfileConfig(
                    adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                    max_batch_tokens=16384,
                ),
            )

    def test_zero_kv_budget_on_gen_profile_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive int"):
            self._make_gen_config(
                ProfileConfig(
                    adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                    max_batch_tokens=16384,
                    kv_budget_tokens=0,
                ),
            )

    def test_negative_kv_budget_on_gen_profile_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive int"):
            self._make_gen_config(
                ProfileConfig(
                    adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                    max_batch_tokens=16384,
                    kv_budget_tokens=-512,
                ),
            )

    def test_positive_kv_budget_accepted_and_resolved(self) -> None:
        config = self._make_gen_config(
            ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
                admission_enabled=False,
            ),
        )
        resolved = config.resolve_profile("default")
        assert resolved.kv_budget_tokens == 8192
        assert resolved.admission_enabled is False

    def test_kv_budget_not_required_on_non_gen_models(self) -> None:
        """Encode-only / score-only / extract-only models keep working."""
        config = ModelConfig(
            sie_id="bge-m3",
            hf_id="BAAI/bge-m3",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=1024))),
            profiles={
                "default": ProfileConfig(adapter_path="mod:Cls", max_batch_tokens=8192),
            },
        )
        resolved = config.resolve_profile("default")
        assert resolved.kv_budget_tokens is None

    def test_child_profile_inherits_kv_budget(self) -> None:
        """Child profile inherits parent's kv_budget_tokens when not set."""
        config = self._make_gen_config(
            ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=4096,
            ),
            extra={
                "fast": ProfileConfig(extends="default", max_batch_tokens=8192),
            },
        )
        resolved = config.resolve_profile("fast")
        assert resolved.kv_budget_tokens == 4096

    def test_child_profile_overrides_kv_budget(self) -> None:
        config = self._make_gen_config(
            ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=4096,
            ),
            extra={
                "big": ProfileConfig(extends="default", kv_budget_tokens=16384),
            },
        )
        resolved = config.resolve_profile("big")
        assert resolved.kv_budget_tokens == 16384

    def test_child_missing_when_parent_missing_rejected(self) -> None:
        """A child that doesn't supply kv_budget_tokens and whose parent
        also lacks it is rejected (parent is then itself rejected first).
        """
        with pytest.raises(ValidationError, match="kv_budget_tokens"):
            self._make_gen_config(
                ProfileConfig(
                    adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                    max_batch_tokens=16384,
                    # No kv_budget_tokens — parent is rejected.
                ),
                extra={"fast": ProfileConfig(extends="default", max_batch_tokens=8192)},
            )

    def test_oversubscribed_budget_emits_warning(self) -> None:
        """Over-subscribed kv_budget_tokens emits a UserWarning, not an error."""
        with pytest.warns(UserWarning, match="kv_budget_tokens"):
            self._make_gen_config(
                ProfileConfig(
                    adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                    max_batch_tokens=16384,
                    # Absurdly large to trip the coarse derivation guard.
                    kv_budget_tokens=10_000_000,
                    adapter_options=AdapterOptions(loadtime={"mem_fraction_static": 0.85}),
                ),
            )

    def test_resolved_profile_carries_admission_fields(self) -> None:
        config = self._make_gen_config(
            ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
                admission_enabled=True,
            ),
        )
        resolved = config.resolve_profile("default")
        assert isinstance(resolved, ResolvedProfile)
        assert resolved.kv_budget_tokens == 8192
        assert resolved.admission_enabled is True


class TestGrammarProfile:
    """``tasks.generate.grammar_profile`` validation (grammar-request routing)."""

    @staticmethod
    def _config(grammar_profile: str | None, *, with_no_spec: bool = True) -> ModelConfig:
        profiles = {
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.test:TestAdapter",
                max_batch_tokens=8192,
                kv_budget_tokens=1024,
            ),
        }
        if with_no_spec:
            profiles["no-spec"] = ProfileConfig(
                adapter_path="sie_server.adapters.test:TestAdapter",
                max_batch_tokens=8192,
                kv_budget_tokens=1024,
            )
        return ModelConfig(
            sie_id="gen-model",
            hf_id="org/gen",
            tasks=Tasks(
                generate=GenerateTask(context_length=8192, max_output_tokens=512, grammar_profile=grammar_profile)
            ),
            profiles=profiles,
        )

    def test_valid_grammar_profile(self) -> None:
        config = self._config("no-spec")
        assert config.tasks.generate.grammar_profile == "no-spec"  # type: ignore[union-attr]

    def test_none_is_default(self) -> None:
        config = self._config(None)
        assert config.tasks.generate.grammar_profile is None  # type: ignore[union-attr]

    def test_unknown_grammar_profile_rejected(self) -> None:
        with pytest.raises(ValidationError, match="grammar_profile"):
            self._config("does-not-exist")

    def test_grammar_profile_default_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be 'default'"):
            self._config("default")


class TestLoraRevisionPins:
    """The pinned-LoRA spelling on loadtime.lora_paths dict values (#2113)."""

    SHA_A = "a" * 40
    SHA_B = "b" * 40

    @staticmethod
    def _profile(loadtime: dict | None = None, runtime: dict | None = None) -> ProfileConfig:
        return ProfileConfig(
            adapter_path="mod:Cls",
            max_batch_tokens=8192,
            adapter_options=AdapterOptions(loadtime=loadtime or {}, runtime=runtime or {}),
        )

    def test_pinned_dict_value_is_accepted_and_surfaced(self) -> None:
        config = _make_config(
            profiles={
                "default": self._profile({"lora_paths": {"bank": {"id": "acme/l", "revision": self.SHA_A}}}),
            }
        )
        assert config.lora_revisions() == {"acme/l": self.SHA_A}

    def test_bare_dict_and_list_values_stay_unpinned(self) -> None:
        config = _make_config(
            profiles={
                "default": self._profile({"lora_paths": {"bank": "acme/l"}}),
                "listed": ProfileConfig(
                    extends="default", adapter_options=AdapterOptions(loadtime={"lora_paths": ["acme/m"]})
                ),
                "legacy": ProfileConfig(
                    extends="default", adapter_options=AdapterOptions(runtime={"lora_id": "acme/n"})
                ),
            }
        )
        assert config.lora_revisions() == {"acme/l": None, "acme/m": None, "acme/n": None}

    def test_branch_name_revision_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="40-char commit SHA"):
            self._profile({"lora_paths": {"bank": {"id": "acme/l", "revision": "main"}}})

    def test_unknown_key_in_pin_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown key"):
            self._profile({"lora_paths": {"bank": {"id": "acme/l", "rev": self.SHA_A}}})

    def test_list_form_stays_bare_only(self) -> None:
        with pytest.raises(ValidationError, match="dict form"):
            self._profile({"lora_paths": [{"id": "acme/l", "revision": self.SHA_A}]})

    def test_conflicting_pins_across_profiles_are_rejected_at_config_load(self) -> None:
        with pytest.raises(ValidationError, match="two different revisions"):
            _make_config(
                profiles={
                    "default": self._profile({"lora_paths": {"x": {"id": "acme/l", "revision": self.SHA_A}}}),
                    "other": ProfileConfig(
                        extends="default",
                        adapter_options=AdapterOptions(
                            loadtime={"lora_paths": {"x": {"id": "acme/l", "revision": self.SHA_B}}}
                        ),
                    ),
                }
            )

    def test_pin_beats_bare_mention_of_the_same_id(self) -> None:
        config = _make_config(
            profiles={
                "default": self._profile({"lora_paths": {"x": {"id": "acme/l", "revision": self.SHA_A}}}),
                "legacy": ProfileConfig(
                    extends="default", adapter_options=AdapterOptions(runtime={"lora_id": "acme/l"})
                ),
            }
        )
        assert config.lora_revisions() == {"acme/l": self.SHA_A}
