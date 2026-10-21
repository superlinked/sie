"""Model ``pool`` filtering in the registry, and mixed task classes in one pool.

A worker's registry keeps only configs whose ``pool`` matches ``SIE_POOL``.
Generation and encoder models may share a pool: a composite bundle runs them
on one GPU, each in its own component process, so the registry no longer
refuses the mix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry


def _gen_config(sie_id: str = "Qwen/Qwen3-4B-Instruct-2507") -> ModelConfig:
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
        },
    )


def _encode_config(sie_id: str = "BAAI/bge-m3") -> ModelConfig:
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=1024))),
        profiles={
            "default": ProfileConfig(adapter_path="mod:Encoder", max_batch_tokens=8192),
        },
    )


def _write_config(models_dir: Path, config: ModelConfig) -> None:
    path = models_dir / f"{config.sie_id.replace('/', '__')}.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True), sort_keys=False))


class TestMixedTaskClasses:
    def test_add_config_accepts_generation_and_encoders_in_one_pool(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        gen = _gen_config()
        gen.pool = "p1"
        enc = _encode_config()
        enc.pool = "p1"

        registry.add_config(gen)
        registry.add_config(enc)

        assert sorted(registry.model_names) == ["BAAI/bge-m3", "Qwen/Qwen3-4B-Instruct-2507"]

    def test_boot_from_dir_accepts_a_mixed_pool(self, tmp_path: Path) -> None:
        _write_config(tmp_path, _encode_config())
        _write_config(tmp_path, _gen_config())

        registry = ModelRegistry(models_dir=tmp_path, pool_name="default")

        assert sorted(registry.model_names) == ["BAAI/bge-m3", "Qwen/Qwen3-4B-Instruct-2507"]

    @pytest.mark.asyncio
    async def test_config_snapshot_mixing_task_classes_applies(self) -> None:
        registry = ModelRegistry(pool_name="default")

        await registry.replace_configs_async([_encode_config(), _gen_config()])

        assert sorted(registry.model_names) == ["BAAI/bge-m3", "Qwen/Qwen3-4B-Instruct-2507"]


class TestModelPoolFilter:
    def test_add_config_skips_different_model_pool(self) -> None:
        registry = ModelRegistry(pool_name="customer-a")
        default_config = _encode_config("default/model")
        tenant_config = _encode_config("tenant/model")
        tenant_config.pool = "customer-a"

        registry.add_config(default_config)
        registry.add_config(tenant_config)

        assert "default/model" not in registry.model_names
        assert "tenant/model" in registry.model_names

    def test_load_configs_from_dir_filters_different_model_pool(self, tmp_path: Path) -> None:
        default_config = _encode_config("default/model")
        tenant_config = _encode_config("tenant/model")
        tenant_config.pool = "Customer-A"
        _write_config(tmp_path, default_config)
        _write_config(tmp_path, tenant_config)

        registry = ModelRegistry(models_dir=tmp_path, pool_name="customer-a")

        assert registry.model_names == ["tenant/model"]

    @pytest.mark.asyncio
    async def test_replace_configs_filters_different_model_pool(self) -> None:
        registry = ModelRegistry(pool_name="customer-a")
        default_config = _encode_config("default/model")
        tenant_config = _encode_config("tenant/model")
        tenant_config.pool = "customer-a"

        await registry.replace_configs_async([default_config, tenant_config])

        assert registry.model_names == ["tenant/model"]
