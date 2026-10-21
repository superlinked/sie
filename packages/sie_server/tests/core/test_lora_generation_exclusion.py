"""Tests for the legacy scalar ``lora_id`` exclusion validator.

Multi-LoRA on generation models *has shipped* — it is declared via
``profile.adapter_options.loadtime["lora_paths"]`` (a list of adapter
ids). The exclusion that remains is the *legacy scalar* spelling
``profile.adapter_options.runtime["lora_id"]``: a pre-Multi-LoRA shape
that no longer maps to a supported code path on the generation primitive
and is rejected at config-load time so the misconfiguration is visible
before request traffic.

The validator runs at config-load and at
:meth:`ModelRegistry.add_config` time, *regardless* of ``SIE_POOL`` — it
is a hard invariant. Encode/score/extract models continue to use the
scalar ``lora_id`` form and are accepted.

This test locks the generation-specific config-load rejection.
"""

from __future__ import annotations

import pytest
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.legacy_lora import (
    LegacyScalarLoraIdError,
    _has_legacy_scalar_lora_id,
    validate_no_legacy_scalar_lora_id,
)
from sie_server.core.registry import ModelRegistry

# --- fixtures -----------------------------------------------------------------


def _gen_config(
    sie_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    *,
    lora_id: str | None = None,
    lora_paths: list[str] | None = None,
) -> ModelConfig:
    runtime: dict[str, object] = {}
    if lora_id is not None:
        runtime["lora_id"] = lora_id
    loadtime: dict[str, object] = {}
    if lora_paths is not None:
        loadtime["lora_paths"] = lora_paths
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
        tasks=Tasks(
            generate=GenerateTask(
                context_length=32768,
                max_output_tokens=4096,
                capabilities=GenerateCapabilities(),
            ),
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
                adapter_options=AdapterOptions(runtime=runtime, loadtime=loadtime),
            ),
        },
    )


def _encode_config(
    sie_id: str = "BAAI/bge-m3",
    *,
    lora_id: str | None = None,
) -> ModelConfig:
    runtime: dict[str, object] = {}
    if lora_id is not None:
        runtime["lora_id"] = lora_id
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=1024))),
        profiles={
            "default": ProfileConfig(
                adapter_path="mod:Encoder",
                max_batch_tokens=8192,
                adapter_options=AdapterOptions(runtime=runtime),
            ),
        },
    )


def _with_pool(config: ModelConfig, pool: str) -> ModelConfig:
    config.pool = pool
    return config


def _gen_config_multi_profile(
    sie_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    *,
    lora_id_on_second: str | None = None,
) -> ModelConfig:
    second_runtime: dict[str, object] = {}
    if lora_id_on_second is not None:
        second_runtime["lora_id"] = lora_id_on_second
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
        tasks=Tasks(
            generate=GenerateTask(
                context_length=32768,
                max_output_tokens=4096,
                capabilities=GenerateCapabilities(),
            ),
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
            ),
            "tuned": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
                adapter_options=AdapterOptions(runtime=second_runtime),
            ),
        },
    )


# --- _has_legacy_scalar_lora_id ----------------------------------------------


class TestHasLegacyScalarLoraId:
    def test_no_lora_returns_false(self) -> None:
        assert _has_legacy_scalar_lora_id(_gen_config()) is False

    def test_scalar_lora_id_on_only_profile_returns_true(self) -> None:
        assert _has_legacy_scalar_lora_id(_gen_config(lora_id="org/qwen-lora")) is True

    def test_empty_string_lora_id_treated_as_absent(self) -> None:
        """Matches historical loader behaviour: ``if lora_id:`` skips empty."""
        cfg = _encode_config(lora_id="")
        assert _has_legacy_scalar_lora_id(cfg) is False

    def test_scalar_lora_id_on_secondary_profile_returns_true(self) -> None:
        cfg = _gen_config_multi_profile(lora_id_on_second="org/qwen-lora")
        assert _has_legacy_scalar_lora_id(cfg) is True

    def test_encode_with_scalar_lora_id_returns_true(self) -> None:
        assert _has_legacy_scalar_lora_id(_encode_config(lora_id="org/bge-lora")) is True


# --- validate_no_legacy_scalar_lora_id ---------------------------------------


class TestValidateNoLegacyScalarLoraId:
    def test_generation_with_scalar_lora_id_rejected(self) -> None:
        cfg = _gen_config(lora_id="org/qwen-lora")
        with pytest.raises(LegacyScalarLoraIdError) as exc:
            validate_no_legacy_scalar_lora_id(name="Qwen/Qwen3-4B-Instruct-2507", config=cfg)
        msg = str(exc.value)
        assert "Qwen/Qwen3-4B-Instruct-2507" in msg
        assert "default" in msg  # offending profile named
        assert "lora_id" in msg

    def test_generation_with_scalar_lora_id_on_secondary_profile_rejected(self) -> None:
        cfg = _gen_config_multi_profile(lora_id_on_second="org/qwen-lora")
        with pytest.raises(LegacyScalarLoraIdError) as exc:
            validate_no_legacy_scalar_lora_id(name="qwen", config=cfg)
        # Only the offending profile is named, not the scalar-free one.
        msg = str(exc.value)
        assert "tuned" in msg
        assert "['tuned']" in msg

    def test_generation_without_scalar_lora_id_accepted(self) -> None:
        validate_no_legacy_scalar_lora_id(
            name="Qwen/Qwen3-4B-Instruct-2507",
            config=_gen_config(),
        )

    def test_non_generation_with_scalar_lora_id_accepted(self) -> None:
        """Encode/score/extract continue to use the scalar ``lora_id`` form."""
        validate_no_legacy_scalar_lora_id(
            name="BAAI/bge-m3",
            config=_encode_config(lora_id="org/bge-lora"),
        )

    def test_non_generation_without_lora_accepted(self) -> None:
        validate_no_legacy_scalar_lora_id(
            name="BAAI/bge-m3",
            config=_encode_config(),
        )


# --- Multi-LoRA generation (shipped path) vs legacy scalar -------------------


class TestMultiLoraVsLegacyScalar:
    """Pin the distinction between the shipped Multi-LoRA path and the
    legacy scalar ``lora_id`` form on the generation primitive.

    Multi-LoRA generation is declared via
    ``adapter_options.loadtime["lora_paths"]`` and must be accepted. The
    legacy scalar ``adapter_options.runtime["lora_id"]`` form must still be
    rejected with :class:`LegacyScalarLoraIdError`.
    """

    def test_multi_lora_loadtime_paths_on_generation_accepted(self) -> None:
        cfg = _gen_config(lora_paths=["adapter-a", "adapter-b"])
        # Must not raise — Multi-LoRA generation has shipped.
        validate_no_legacy_scalar_lora_id(
            name="Qwen/Qwen3-4B-Instruct-2507",
            config=cfg,
        )

    def test_legacy_scalar_lora_id_on_generation_rejected(self) -> None:
        cfg = _gen_config(lora_id="adapter-a")
        with pytest.raises(LegacyScalarLoraIdError):
            validate_no_legacy_scalar_lora_id(
                name="Qwen/Qwen3-4B-Instruct-2507",
                config=cfg,
            )


# --- registry wire-in --------------------------------------------------------


class TestRegistryHook:
    def test_add_config_generation_with_scalar_lora_id_rejects(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        with pytest.raises(LegacyScalarLoraIdError):
            registry.add_config(_with_pool(_gen_config(lora_id="org/qwen-lora"), "p1"))

    def test_add_config_generation_without_scalar_lora_id_accepted(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        registry.add_config(_with_pool(_gen_config(), "p1"))

    def test_add_config_encode_with_scalar_lora_id_accepted(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        registry.add_config(_with_pool(_encode_config(lora_id="org/bge-lora"), "p1"))

    def test_validator_fires_without_pool_name(self) -> None:
        """Legacy scalar check is not pool-scoped — fires when SIE_POOL unset."""
        registry = ModelRegistry()  # no pool_name
        with pytest.raises(LegacyScalarLoraIdError):
            registry.add_config(_gen_config(lora_id="org/qwen-lora"))

    def test_failed_add_does_not_mutate_state(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        try:
            registry.add_config(_with_pool(_gen_config(lora_id="org/qwen-lora"), "p1"))
        except LegacyScalarLoraIdError:
            pass
        assert "Qwen/Qwen3-4B-Instruct-2507" not in registry.model_names
