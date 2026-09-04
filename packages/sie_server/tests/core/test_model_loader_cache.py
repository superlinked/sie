import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sie_sdk.cache import CacheConfig
from sie_server.config.model import AdapterOptions, ModelConfig, ProfileConfig, Tasks
from sie_server.config.serving_artifacts import ServingArtifactManifest, canonical_manifest_bytes
from sie_server.core.model_loader import ModelLoader
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.preprocessor_registry import PreprocessorRegistry


@pytest.mark.parametrize(
    ("base_source", "expected_refs"),
    [
        pytest.param(
            {"hf_id": "acme/base", "hf_revision": "a" * 40},
            [("acme/base", "a" * 40), ("acme/draft", "b" * 40)],
            id="hf-base",
        ),
        pytest.param(
            {"weights_path": Path("/models/base")},
            [("acme/draft", "b" * 40)],
            id="local-base",
        ),
    ],
)
def test_ensure_weights_cached_stages_pinned_speculative_draft(
    base_source: dict[str, object],
    expected_refs: list[tuple[str, str]],
) -> None:
    draft_revision = "b" * 40
    config = ModelConfig(
        sie_id="acme/base",
        **base_source,
        tasks=Tasks(),
        profiles={
            "default": ProfileConfig(
                adapter_path="mod:Cls",
                max_batch_tokens=8192,
                adapter_options=AdapterOptions(
                    loadtime={
                        "speculative": {
                            "enabled": True,
                            "algorithm": "nextn",
                            "draft_model": "acme/draft",
                            "draft_model_revision": draft_revision,
                        }
                    }
                ),
            )
        },
    )
    cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-cpu")
    loader = ModelLoader(
        preprocessor_registry=PreprocessorRegistry(),
        postprocessor_registry=PostprocessorRegistry(cpu_pool),
        all_configs={},
    )
    cache_config = MagicMock()

    try:
        with (
            patch("sie_sdk.cache.get_cache_config", return_value=cache_config),
            patch("sie_sdk.cache.ensure_model_cached", return_value=Path("/cache/model")) as ensure,
        ):
            loader.ensure_weights_cached(config)

        assert [(entry.args[0], entry.kwargs["revision"]) for entry in ensure.call_args_list] == expected_refs
        assert all(entry.args[1] is cache_config for entry in ensure.call_args_list)
    finally:
        loader._load_executor.shutdown(wait=True)
        cpu_pool.shutdown(wait=True)


def test_ensure_weights_cached_stages_derived_repo_instead_of_source(tmp_path: Path) -> None:
    source_revision = "a" * 40
    derived_revision = "b" * 40
    config = ModelConfig(
        sie_id="google/source-model",
        hf_id="google/source-model",
        hf_revision=source_revision,
        tasks=Tasks(),
        profiles={
            "default": ProfileConfig(
                adapter_path="mod:Cls",
                max_batch_tokens=128,
                adapter_options=AdapterOptions(
                    loadtime={
                        "serving_artifact": {
                            "format": "ctranslate2",
                            "repo_id": "superlinked/derived-model",
                            "revision": derived_revision,
                            "manifest_path": "sie-serving-artifact.json",
                            "manifest_sha256": "c" * 64,
                            "compute_type": "bfloat16",
                        }
                    }
                ),
            )
        },
    )
    cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-cpu")
    disk_cache = MagicMock()
    loader = ModelLoader(
        preprocessor_registry=PreprocessorRegistry(),
        postprocessor_registry=PostprocessorRegistry(cpu_pool),
        all_configs={},
        disk_cache_manager=disk_cache,
    )
    cache_config = SimpleNamespace(local_cache=tmp_path / "hub")
    cached_repo = tmp_path / "hub" / "models--superlinked--derived-model"
    verified = SimpleNamespace(
        root=tmp_path / "materialized" / ("c" * 64),
        manifest_sha256="c" * 64,
        repo_id="superlinked/derived-model",
        revision=derived_revision,
        compute_type="bfloat16",
    )

    try:
        with (
            patch("sie_sdk.cache.get_cache_config", return_value=cache_config),
            patch("sie_sdk.cache.ensure_model_cached", return_value=cached_repo) as ensure,
            patch(
                "sie_server.core.model_loader.verify_and_materialize_serving_artifact",
                return_value=verified,
            ) as materialize,
        ):
            loader.ensure_weights_cached(config)

        ensure.assert_called_once_with(
            "superlinked/derived-model",
            cache_config,
            revision=derived_revision,
        )
        materialize.assert_called_once()
        assert materialize.call_args.kwargs == {
            "source_hf_id": "google/source-model",
            "source_hf_revision": source_revision,
            "cached_repo_root": cached_repo,
            "materialized_cache_root": cache_config.local_cache / "sie-serving-artifacts-v1",
        }
        assert loader._verified_serving_artifacts[config.sie_id] is verified
        disk_cache.ensure_space_before_download.assert_called_once_with("superlinked/derived-model")
        disk_cache.touch.assert_called_once_with("superlinked/derived-model")
    finally:
        loader._load_executor.shutdown(wait=True)
        cpu_pool.shutdown(wait=True)


def test_unregister_clears_only_in_memory_serving_artifact_registration(tmp_path: Path) -> None:
    cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-cpu")
    preprocessor_registry = PreprocessorRegistry()
    postprocessor_registry = PostprocessorRegistry(cpu_pool)
    loader = ModelLoader(
        preprocessor_registry=preprocessor_registry,
        postprocessor_registry=postprocessor_registry,
        all_configs={},
    )
    artifact_root = tmp_path / "materialized" / ("c" * 64)
    artifact_root.mkdir(parents=True)
    loader._verified_serving_artifacts["google/source-model"] = MagicMock(root=artifact_root)

    try:
        loader.unregister("google/source-model", "cpu")

        assert "google/source-model" not in loader._verified_serving_artifacts
        assert artifact_root.is_dir()
    finally:
        loader._load_executor.shutdown(wait=True)
        cpu_pool.shutdown(wait=True)


def test_offline_cache_only_derived_snapshot_materializes_without_hub_fallback(tmp_path: Path) -> None:
    source_revision = "a" * 40
    derived_revision = "b" * 40
    hub = tmp_path / "hub"
    snapshot = hub / "models--superlinked--derived-model" / "snapshots" / derived_revision
    snapshot.mkdir(parents=True)
    files = {
        "config.json": (b'{"model_type":"Transformer"}\n', "model.config"),
        "model.bin": (b"converted", "model.weights"),
        "tokenizer_config.json": (b'{"tokenizer_class":"T5Tokenizer"}\n', "tokenizer.config"),
        "spiece.model": (b"sentencepiece", "tokenizer.sentencepiece"),
    }
    artifacts = []
    for relative, (content, role) in files.items():
        (snapshot / relative).write_bytes(content)
        artifacts.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = ServingArtifactManifest.model_validate(
        {
            "schema": "sie-ctranslate2-serving-artifact-v1",
            "source": {"hf_id": "google/source-model", "hf_revision": source_revision},
            "converter": {
                "name": "ct2-transformers-converter",
                "version": "4.8.1",
                "torch_version": "2.9.1",
                "transformers_version": "4.57.6",
                "compute_type": "bfloat16",
                "recipe_sha256": "c" * 64,
            },
            "runtime": {"minimum_version": "4.8.1", "maximum_version": "4.8.1"},
            "tokenizer": {
                "class_name": "T5Tokenizer",
                "config_sha256": hashlib.sha256(files["tokenizer_config.json"][0]).hexdigest(),
                "files": [
                    {"path": "spiece.model", "role": "tokenizer.sentencepiece"},
                    {"path": "tokenizer_config.json", "role": "tokenizer.config"},
                ],
            },
            "artifacts": artifacts,
        }
    )
    manifest_bytes = canonical_manifest_bytes(manifest)
    (snapshot / "sie-serving-artifact.json").write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    config = ModelConfig(
        sie_id="google/source-model",
        hf_id="google/source-model",
        hf_revision=source_revision,
        tasks=Tasks(),
        profiles={
            "default": ProfileConfig(
                adapter_path="mod:Cls",
                max_batch_tokens=128,
                adapter_options=AdapterOptions(
                    loadtime={
                        "serving_artifact": {
                            "format": "ctranslate2",
                            "repo_id": "superlinked/derived-model",
                            "revision": derived_revision,
                            "manifest_path": "sie-serving-artifact.json",
                            "manifest_sha256": manifest_sha256,
                            "compute_type": "bfloat16",
                        }
                    }
                ),
            )
        },
    )
    cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-cpu")
    loader = ModelLoader(
        preprocessor_registry=PreprocessorRegistry(),
        postprocessor_registry=PostprocessorRegistry(cpu_pool),
        all_configs={},
    )

    try:
        with (
            patch(
                "sie_sdk.cache.get_cache_config",
                return_value=CacheConfig(local_cache=hub, cluster_cache=None, hf_fallback=False),
            ),
            patch("huggingface_hub.snapshot_download") as hub_download,
        ):
            loader.ensure_weights_cached(config)

        hub_download.assert_not_called()
        verified = loader._verified_serving_artifacts[config.sie_id]
        assert verified.root == hub / "sie-serving-artifacts-v1" / manifest_sha256
        assert (verified.root / "model.bin").read_bytes() == b"converted"
    finally:
        loader._load_executor.shutdown(wait=True)
        cpu_pool.shutdown(wait=True)
