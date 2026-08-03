import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sie_server.adapters.sentence_transformer import SentenceTransformerDenseAdapter
from sie_server.config.engine import ComputePrecision
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    ExtractTask,
    ModelConfig,
    ProfileConfig,
    ScoreTask,
    Tasks,
)
from sie_server.core.loader import (
    _build_adapter_kwargs,
    load_adapter,
    load_model_config,
    load_model_configs,
    resolve_adapter_path,
)


def _make_config(
    sie_id: str = "test",
    hf_id: str | None = "org/test",
    adapter_path: str = "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
    max_batch_tokens: int = 8192,
    dense_dim: int | None = 768,
    sparse_dim: int | None = None,
    multivector_dim: int | None = None,
    score: bool = False,
    extract: bool = False,
    max_sequence_length: int | None = None,
    weights_path: Path | None = None,
    compute_precision: ComputePrecision | None = None,
    adapter_options: AdapterOptions | None = None,
) -> ModelConfig:
    encode = None
    if any(dim is not None for dim in (dense_dim, sparse_dim, multivector_dim)):
        encode = EncodeTask(
            dense=EmbeddingDim(dim=dense_dim) if dense_dim is not None else None,
            sparse=EmbeddingDim(dim=sparse_dim) if sparse_dim is not None else None,
            multivector=EmbeddingDim(dim=multivector_dim) if multivector_dim is not None else None,
        )
    profile = ProfileConfig(
        adapter_path=adapter_path,
        max_batch_tokens=max_batch_tokens,
        compute_precision=compute_precision,
        adapter_options=adapter_options or AdapterOptions(),
    )
    return ModelConfig(
        sie_id=sie_id,
        hf_id=hf_id,
        weights_path=weights_path,
        tasks=Tasks(
            encode=encode,
            score=ScoreTask() if score else None,
            extract=ExtractTask() if extract else None,
        ),
        profiles={"default": profile},
        max_sequence_length=max_sequence_length,
    )


class TestLoadModelConfig:
    """Tests for load_model_config."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Can load a valid config file."""
        config_path = tmp_path / "test-model.yaml"
        config_path.write_text("""
sie_id: test-model
hf_id: org/test-model
tasks:
  encode:
    dense:
      dim: 768
max_sequence_length: 512
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
""")

        config = load_model_config(config_path)

        assert config.sie_id == "test-model"
        assert config.hf_id == "org/test-model"
        assert config.tasks.encode.dense is not None
        assert config.tasks.encode.dense.dim == 768

    def test_load_missing_config(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing config."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_model_config(config_path)


class TestLoadModelConfigs:
    """Tests for load_model_configs."""

    def test_load_multiple_configs(self, tmp_path: Path) -> None:
        """Can load multiple configs from a directory."""
        (tmp_path / "model-a.yaml").write_text("""
sie_id: model-a
hf_id: org/model-a
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
""")

        (tmp_path / "model-b.yaml").write_text("""
sie_id: model-b
hf_id: org/model-b
tasks:
  encode:
    sparse:
      dim: 30522
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerSparseAdapter"
    max_batch_tokens: 8192
""")

        configs = load_model_configs(tmp_path)

        assert len(configs) == 2
        assert "model-a" in configs
        assert "model-b" in configs
        assert configs["model-a"].tasks.encode.dense is not None
        assert configs["model-a"].tasks.encode.dense.dim == 384
        assert configs["model-b"].tasks.encode.sparse is not None
        assert configs["model-b"].tasks.encode.sparse.dim == 30522

    def test_skip_directories_without_config(self, tmp_path: Path) -> None:
        """Skips directories and non-yaml files."""
        (tmp_path / "valid-model.yaml").write_text("""
sie_id: valid-model
hf_id: org/valid
tasks:
  encode:
    dense:
      dim: 768
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
""")

        # Directories should be ignored (old format)
        (tmp_path / "empty-dir").mkdir()

        configs = load_model_configs(tmp_path)

        assert len(configs) == 1
        assert "valid-model" in configs

    def test_profile_max_seq_length_is_promoted_to_variant_context(self, tmp_path: Path) -> None:
        """Profile variants can expose a larger generation context than the base model."""
        (tmp_path / "qwen.yaml").write_text("""
sie_id: Qwen/Qwen3.6-27B
hf_id: Qwen/Qwen3.6-27B
tasks:
  generate:
    context_length: 4096
    max_output_tokens: 4096
max_sequence_length: 4096
profiles:
  default:
    adapter_path: "sie_server.adapters.sglang.generation:SGLangGenerationAdapter"
    max_batch_tokens: 16384
    kv_budget_tokens: 8192
    admission_enabled: true
    adaptive_batching:
      target_p50_ms: 250
      min_wait_ms: 2
  rtx-pro-6000:
    adapter_path: "sie_server.adapters.sglang.generation:SGLangGenerationAdapter"
    max_batch_tokens: 32768
    kv_budget_tokens: 65536
    adapter_options:
      loadtime:
        max_seq_length: 32768
  long-startup:
    adapter_path: "sie_server.adapters.sglang.generation:SGLangGenerationAdapter"
    max_batch_tokens: 32768
    kv_budget_tokens: 32768
    adapter_options:
      loadtime:
        max_seq_length: 32768
        startup_timeout_s: 900
        extra_launch_args:
          - "--example-flag"
  extended-startup:
    extends: default
    adapter_options:
      loadtime:
        max_seq_length: 32768
        startup_timeout_s: 1200
""")

        configs = load_model_configs(tmp_path)

        base = configs["Qwen/Qwen3.6-27B"]
        variant = configs["Qwen/Qwen3.6-27B:rtx-pro-6000"]
        long_startup_variant = configs["Qwen/Qwen3.6-27B:long-startup"]
        extended_startup_variant = configs["Qwen/Qwen3.6-27B:extended-startup"]
        assert base.tasks.generate is not None
        assert variant.tasks.generate is not None
        assert long_startup_variant.tasks.generate is not None
        assert extended_startup_variant.tasks.generate is not None
        assert base.tasks.generate.context_length == 4096
        assert base.max_sequence_length == 4096
        assert variant.tasks.generate.context_length == 32768
        assert variant.max_sequence_length == 32768
        assert long_startup_variant.tasks.generate.context_length == 32768
        assert long_startup_variant.max_sequence_length == 32768
        assert extended_startup_variant.tasks.generate.context_length == 32768
        assert extended_startup_variant.max_sequence_length == 32768
        assert _build_adapter_kwargs(variant, "bfloat16")["max_seq_length"] == 32768
        long_startup_kwargs = _build_adapter_kwargs(long_startup_variant, "bfloat16")
        assert long_startup_kwargs["max_seq_length"] == 32768
        assert long_startup_kwargs["startup_timeout_s"] == 900
        assert long_startup_kwargs["extra_launch_args"] == ["--example-flag"]
        extended_startup_kwargs = _build_adapter_kwargs(extended_startup_variant, "bfloat16")
        assert extended_startup_kwargs["max_seq_length"] == 32768
        assert extended_startup_kwargs["startup_timeout_s"] == 1200
        extended_profile = extended_startup_variant.resolve_profile("default")
        assert extended_profile.kv_budget_tokens == 8192
        assert extended_profile.admission_enabled is True
        assert extended_profile.adaptive_batching is not None
        assert extended_profile.adaptive_batching.target_p50_ms == 250
        assert extended_profile.adaptive_batching.min_wait_ms == 2
        assert extended_startup_variant._resolved_cache is not base._resolved_cache
        assert extended_startup_variant._resolved_cache is not variant._resolved_cache
        assert extended_startup_variant._resolved_cache is not long_startup_variant._resolved_cache
        assert extended_startup_variant._resolved_lock is not base._resolved_lock
        assert extended_startup_variant._resolved_lock is not variant._resolved_lock
        assert extended_startup_variant._resolved_lock is not long_startup_variant._resolved_lock

    def test_missing_models_dir(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing models directory."""
        missing_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Models directory not found"):
            load_model_configs(missing_dir)


class TestResolveAdapterPath:
    """Tests for resolve_adapter_path."""

    def test_builtin_adapter(self, tmp_path: Path) -> None:
        """Resolves built-in adapter path."""
        config = _make_config(
            sie_id="test",
            hf_id="org/test",
            adapter_path="sie_server.adapters.bge_m3:BGEM3Adapter",
            dense_dim=1024,
            sparse_dim=250002,
            multivector_dim=1024,
        )

        path = resolve_adapter_path(config, tmp_path)

        assert path == "sie_server.adapters.bge_m3:BGEM3Adapter"

    def test_custom_adapter_file(self, tmp_path: Path) -> None:
        """Resolves custom adapter file path."""
        config = _make_config(
            sie_id="custom",
            hf_id="org/custom",
            adapter_path="adapter.py:CustomAdapter",
        )

        path = resolve_adapter_path(config, tmp_path)

        assert path == f"{tmp_path / 'adapter.py'}:CustomAdapter"

    def test_invalid_adapter_path(self, tmp_path: Path) -> None:
        """Raises error for adapter path without colon."""
        config = _make_config(
            sie_id="test",
            hf_id="org/test",
            adapter_path="invalid_no_colon",
        )

        # The adapter_path "invalid_no_colon" doesn't start with "sie_server."
        # and has no colon, so it raises ValueError
        with pytest.raises(ValueError, match="Invalid adapter path"):
            resolve_adapter_path(config, tmp_path)


class TestBuildAdapterKwargs:
    """Tests for _build_adapter_kwargs."""

    def test_with_hf_id(self) -> None:
        """Builds kwargs with HF model ID."""
        config = _make_config(
            hf_id="org/test-model",
            max_sequence_length=512,
            adapter_options=AdapterOptions(loadtime={"normalize": True}),
        )

        kwargs = _build_adapter_kwargs(config, "float16")

        assert kwargs["model_name_or_path"] == "org/test-model"
        assert kwargs["normalize"] is True
        assert kwargs["max_seq_length"] == 512
        assert kwargs["compute_precision"] == "float16"

    def test_weights_path_precedence(self) -> None:
        """weights_path takes precedence over hf_id."""
        config = _make_config(
            hf_id="org/test-model",
            weights_path=Path("/data/models/test"),
        )

        kwargs = _build_adapter_kwargs(config, "float16")

        assert kwargs["model_name_or_path"] == Path("/data/models/test")

    def test_model_precision_override(self) -> None:
        """Model config precision overrides default."""
        config = _make_config(
            hf_id="org/test-model",
            compute_precision="bfloat16",
        )

        kwargs = _build_adapter_kwargs(config, "float16")

        # Model's bfloat16 should override default float16
        assert kwargs["compute_precision"] == "bfloat16"

    def test_package_backed_passes_none_for_model_path(self) -> None:
        """package_backed adapters get model_name_or_path=None — they ship their own weights."""
        profile = ProfileConfig(
            adapter_path="sie_server.adapters.docling.adapter:DoclingAdapter",
            max_batch_tokens=1,
        )
        config = ModelConfig(
            sie_id="docling",
            package_backed=True,
            tasks=Tasks(extract=ExtractTask()),
            profiles={"default": profile},
        )

        kwargs = _build_adapter_kwargs(config, "float16")

        assert kwargs["model_name_or_path"] is None

    def test_package_backed_staged_manifest_is_verified_and_passed_to_adapter(self, tmp_path: Path) -> None:
        artifact = tmp_path / "layout" / "model.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"model")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "artifacts": [
                        {
                            "path": "layout/model.bin",
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            "size_bytes": artifact.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        config = ModelConfig(
            sie_id="package/model",
            package_backed=True,
            tasks=Tasks(extract=ExtractTask()),
            profiles={
                "default": ProfileConfig(
                    adapter_path="test:Adapter",
                    max_batch_tokens=1,
                    adapter_options=AdapterOptions(
                        loadtime={
                            "package_artifact_mode": "staged",
                            "package_artifact_manifest_path": str(manifest),
                            "package_artifact_manifest_sha256": manifest_sha256,
                        }
                    ),
                )
            },
        )

        kwargs = _build_adapter_kwargs(config, "float16")

        assert kwargs["package_artifact_root"] == tmp_path
        assert kwargs["package_artifact_manifest_sha256"] == manifest_sha256
        assert "package_artifact_mode" not in kwargs
        assert "package_artifact_manifest_path" not in kwargs

    def test_package_backed_live_downloads_are_rejected_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = ModelConfig(
            sie_id="package/model",
            package_backed=True,
            tasks=Tasks(extract=ExtractTask()),
            profiles={
                "default": ProfileConfig(
                    adapter_path="test:Adapter",
                    max_batch_tokens=1,
                    adapter_options=AdapterOptions(loadtime={"package_artifact_mode": "live"}),
                )
            },
        )
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        with pytest.raises(ValueError, match="live external downloads"):
            _build_adapter_kwargs(config, "float16")


class TestLoadAdapter:
    """Tests for load_adapter."""

    def test_load_builtin_adapter(self, tmp_path: Path) -> None:
        """Can load a built-in adapter."""
        config = _make_config(
            sie_id="test-dense",
            hf_id="sentence-transformers/all-MiniLM-L6-v2",
            dense_dim=384,
            max_sequence_length=256,
            adapter_options=AdapterOptions(loadtime={"normalize": True}),
        )

        # Mock the actual SentenceTransformer to avoid downloading
        with patch("sie_server.adapters.sentence_transformer.SentenceTransformer"):
            adapter = load_adapter(config, tmp_path, device="cpu")

        # Should be the right type
        assert isinstance(adapter, SentenceTransformerDenseAdapter)

    def test_load_bge_m3_flag_variant_with_dense_dim(self, tmp_path: Path) -> None:
        """BGE-M3 Flag adapter accepts dense_dim supplied by loader."""
        config = _make_config(
            sie_id="BAAI/bge-m3:bge_m3_flag",
            hf_id="BAAI/bge-m3",
            adapter_path="sie_server.adapters.bge_m3_flag:BGEM3FlagAdapter",
            dense_dim=1024,
            sparse_dim=250002,
            multivector_dim=1024,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "BGEM3FlagAdapter"

    def test_load_qwen3_vl_embedding_accepts_dense_dim(self, tmp_path: Path) -> None:
        """Qwen3 VL embedding adapter accepts dense_dim supplied by loader."""
        config = _make_config(
            sie_id="Qwen/Qwen3-VL-Embedding-2B",
            hf_id="Qwen/Qwen3-VL-Embedding-2B",
            adapter_path="sie_server.adapters.qwen3_vl_embedding:Qwen3VLEmbeddingAdapter",
            dense_dim=2048,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "Qwen3VLEmbeddingAdapter"
        assert adapter.dims.dense == 2048

    def test_load_custom_adapter(self, tmp_path: Path) -> None:
        """Can load a custom adapter from file."""
        # Create custom adapter file
        adapter_file = tmp_path / "custom_adapter.py"
        adapter_file.write_text("""
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.types.inputs import Item
from typing import Any

class MyCustomAdapter(ModelAdapter):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_path = model_name_or_path
        self.kwargs = kwargs

    @property
    def capabilities(self):
        return ModelCapabilities(inputs=["text"], outputs=["dense"])

    @property
    def dims(self):
        return ModelDims(dense=768)

    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass
""")

        config = _make_config(
            sie_id="custom",
            hf_id="org/custom",
            adapter_path="custom_adapter.py:MyCustomAdapter",
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert adapter.model_path == "org/custom"  # type: ignore

    def test_runtime_default_sampling_wired_when_adapter_accepts_it(self, tmp_path: Path) -> None:
        """A profile's ``runtime.default_sampling`` reaches an adapter that declares the kwarg.

        Regression: the block lived under ``adapter_options.runtime`` and was
        dropped by ``_build_adapter_kwargs`` (which only spreads ``loadtime``),
        so e.g. a ``min_new_tokens`` floor never reached SGLang.
        """
        adapter_file = tmp_path / "ds_adapter.py"
        adapter_file.write_text("""
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims

class DSAdapter(ModelAdapter):
    def __init__(self, model_name_or_path, *, default_sampling=None, **kwargs):
        self.model_path = model_name_or_path
        self.default_sampling = default_sampling

    @property
    def capabilities(self):
        return ModelCapabilities(inputs=["text"], outputs=["dense"])

    @property
    def dims(self):
        return ModelDims(dense=768)

    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass
""")
        config = _make_config(
            sie_id="ds-model",
            hf_id="org/ds",
            adapter_path="ds_adapter.py:DSAdapter",
            adapter_options=AdapterOptions(runtime={"default_sampling": {"min_new_tokens": 10, "temperature": 0.7}}),
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert adapter.default_sampling == {"min_new_tokens": 10, "temperature": 0.7}  # type: ignore

    def test_runtime_default_sampling_skipped_when_adapter_lacks_kwarg(self, tmp_path: Path) -> None:
        """Adapters without a ``default_sampling`` parameter are unaffected (no kwarg leak)."""
        adapter_file = tmp_path / "plain_adapter.py"
        adapter_file.write_text("""
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims

class PlainAdapter(ModelAdapter):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_path = model_name_or_path
        self.seen_kwargs = kwargs

    @property
    def capabilities(self):
        return ModelCapabilities(inputs=["text"], outputs=["dense"])

    @property
    def dims(self):
        return ModelDims(dense=768)

    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass
""")
        config = _make_config(
            sie_id="plain-model",
            hf_id="org/plain",
            adapter_path="plain_adapter.py:PlainAdapter",
            adapter_options=AdapterOptions(runtime={"default_sampling": {"min_new_tokens": 10}}),
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        # **kwargs is not a named ``default_sampling`` parameter, so the guard
        # skips it rather than leaking an unexpected kwarg.
        assert "default_sampling" not in adapter.seen_kwargs  # type: ignore

    def test_invalid_adapter_path(self, tmp_path: Path) -> None:
        """Raises error for invalid adapter path."""
        config = _make_config(
            sie_id="test",
            hf_id="org/test",
            adapter_path="invalid_no_colon",
        )

        with pytest.raises(ValueError, match="Invalid adapter path"):
            load_adapter(config, tmp_path, device="cpu")


class TestAdapterFactoryMethodSwapping:
    """Tests for loader-level device-aware adapter swapping via factory methods."""

    def test_flash_cross_encoder_swaps_to_regular_on_cpu(self, tmp_path: Path) -> None:
        """Loader returns CrossEncoderAdapter when flash adapter is used on CPU."""
        config = _make_config(
            sie_id="test-flash-cross",
            hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            adapter_path="sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            dense_dim=None,
            score=True,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        # Verify we got the fallback, not the flash version
        assert type(adapter).__name__ == "CrossEncoderAdapter"
        assert type(adapter).__name__ != "BertFlashCrossEncoderAdapter"

    def test_flash_cross_encoder_stays_on_cuda(self, tmp_path: Path) -> None:
        """Loader returns BertFlashCrossEncoderAdapter on CUDA when flash-attn is installed."""
        pytest.importorskip("flash_attn", reason="flash-attn not installed")

        config = _make_config(
            sie_id="test-flash-cross",
            hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            adapter_path="sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            dense_dim=None,
            score=True,
        )

        with patch("sie_server.core.inference.is_flash_attention_available", return_value=True):
            adapter = load_adapter(config, tmp_path, device="cuda:0")

        assert type(adapter).__name__ == "BertFlashCrossEncoderAdapter"

    def test_flash_cross_encoder_swaps_on_mps(self, tmp_path: Path) -> None:
        """Loader returns CrossEncoderAdapter when flash adapter is used on MPS."""
        config = _make_config(
            sie_id="test-flash-cross",
            hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            adapter_path="sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            dense_dim=None,
            score=True,
        )

        adapter = load_adapter(config, tmp_path, device="mps")

        assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_flash_adapter_swaps_when_flash_attn_not_installed(self, tmp_path: Path) -> None:
        """Loader returns fallback when flash-attn import fails on CUDA."""
        config = _make_config(
            sie_id="test-flash-cross",
            hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            adapter_path="sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            dense_dim=None,
            score=True,
        )

        # Mock flash_attn import to fail
        with patch.dict(sys.modules, {"flash_attn": None}):
            adapter = load_adapter(config, tmp_path, device="cuda:0")

        # Should fallback even on CUDA if flash-attn not available
        assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_dense_flash_adapter_swaps_to_sentence_transformer_on_cpu(self, tmp_path: Path) -> None:
        """Dense flash adapters fallback to SentenceTransformerDenseAdapter on CPU."""
        config = _make_config(
            sie_id="test-flash-dense",
            hf_id="intfloat/e5-base-v2",
            adapter_path="sie_server.adapters.bert_flash:BertFlashAdapter",
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"

    def test_bge_m3_flash_swaps_to_bge_m3_on_mps(self, tmp_path: Path) -> None:
        """BGE-M3 flash adapter falls back to BGEM3Adapter on MPS."""
        config = _make_config(
            sie_id="test-bge-m3-flash",
            hf_id="BAAI/bge-m3",
            adapter_path="sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter",
            dense_dim=1024,
            sparse_dim=250002,
            multivector_dim=1024,
        )

        adapter = load_adapter(config, tmp_path, device="mps")

        assert type(adapter).__name__ == "BGEM3Adapter"

    def test_colbert_flash_swaps_to_colbert_on_cpu(self, tmp_path: Path) -> None:
        """ColBERT flash adapters fall back to ColBERTAdapter on CPU."""
        config = _make_config(
            sie_id="test-colbert-flash",
            hf_id="jinaai/jina-colbert-v2",
            adapter_path="sie_server.adapters.colbert_rotary_flash:ColBERTRotaryFlashAdapter",
            dense_dim=None,
            multivector_dim=128,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "ColBERTAdapter"

    def test_concurrent_flash_adapter_loading_different_devices(self, tmp_path: Path) -> None:
        """Test concurrent loading of flash adapters on different devices is thread-safe."""
        import concurrent.futures

        config = _make_config(
            sie_id="test-flash-cross",
            hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            adapter_path="sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            dense_dim=None,
            score=True,
        )

        # Load same adapter on different devices concurrently
        devices = ["cpu", "mps", "cpu", "mps"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_adapter, config, tmp_path, device=device) for device in devices]
            adapters = [future.result() for future in futures]

        # All should return CrossEncoderAdapter (fallback on non-CUDA)
        for adapter in adapters:
            assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_concurrent_flash_adapter_loading_same_device(self, tmp_path: Path) -> None:
        """Test concurrent loading of same flash adapter on same device is thread-safe."""
        import concurrent.futures

        config = _make_config(
            sie_id="test-flash-cross",
            hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            adapter_path="sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            dense_dim=None,
            score=True,
        )

        # Load same adapter on same device concurrently (simulates multiple requests)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(load_adapter, config, tmp_path, device="cpu") for _ in range(3)]
            adapters = [future.result() for future in futures]

        # All should return CrossEncoderAdapter (fallback)
        for adapter in adapters:
            assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_concurrent_mixed_adapter_loading(self, tmp_path: Path) -> None:
        """Test concurrent loading of different adapter types (flash and non-flash)."""
        import concurrent.futures

        flash_config = _make_config(
            sie_id="test-flash",
            hf_id="intfloat/e5-base-v2",
            adapter_path="sie_server.adapters.bert_flash:BertFlashAdapter",
        )

        regular_config = _make_config(
            sie_id="test-regular",
            hf_id="intfloat/e5-base-v2",
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Mix flash and regular adapters
            futures = [
                executor.submit(load_adapter, flash_config, tmp_path, device="cpu"),
                executor.submit(load_adapter, regular_config, tmp_path, device="cpu"),
                executor.submit(load_adapter, flash_config, tmp_path, device="mps"),
                executor.submit(load_adapter, regular_config, tmp_path, device="mps"),
            ]
            adapters = [future.result() for future in futures]

        # All flash adapters should fallback to SentenceTransformerDenseAdapter
        # Regular adapters should stay as SentenceTransformerDenseAdapter
        for adapter in adapters:
            assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"
