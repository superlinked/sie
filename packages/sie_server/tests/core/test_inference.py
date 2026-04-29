"""Tests for inference optimization utilities."""

import pytest
import torch
from sie_server.core.inference import (
    InferenceSettings,
    get_torch_dtype,
    is_sdpa_available,
    resolve_attention_backend,
    resolve_compute_precision,
    resolve_inference_settings,
)


class TestGetTorchDtype:
    """Tests for get_torch_dtype function."""

    def test_float16(self) -> None:
        assert get_torch_dtype("float16") == torch.float16

    def test_bfloat16(self) -> None:
        assert get_torch_dtype("bfloat16") == torch.bfloat16

    def test_float32(self) -> None:
        assert get_torch_dtype("float32") == torch.float32


class TestSdpaAvailable:
    """Tests for SDPA availability check."""

    def test_sdpa_available_pytorch2(self) -> None:
        # SDPA is available in PyTorch 2.0+
        # This test will pass on any modern PyTorch installation
        torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        expected = torch_version >= (2, 0)
        assert is_sdpa_available() == expected


class TestResolveAttentionBackend:
    """Tests for attention backend resolution."""

    def test_auto_selects_flash_on_cuda(self) -> None:
        # On CUDA with flash attention available, should select FA2
        backend = resolve_attention_backend("auto", "float16", "cuda:0")
        # Will be either flash_attention_2 (if available) or sdpa
        assert backend in ("flash_attention_2", "sdpa", "eager")

    def test_auto_selects_sdpa_on_cpu(self) -> None:
        # On CPU, should never select flash attention
        backend = resolve_attention_backend("auto", "float16", "cpu")
        assert backend in ("sdpa", "eager")

    def test_fp32_cannot_use_flash_attention(self) -> None:
        # FP32 is not compatible with Flash Attention 2
        backend = resolve_attention_backend("flash_attention_2", "float32", "cuda:0")
        # Should fall back to sdpa or eager
        assert backend in ("sdpa", "eager")

    def test_flash_attention_on_cpu_falls_back(self) -> None:
        # Flash Attention 2 requires CUDA, should fall back on CPU
        backend = resolve_attention_backend("flash_attention_2", "float16", "cpu")
        assert backend in ("sdpa", "eager")

    def test_explicit_eager(self) -> None:
        backend = resolve_attention_backend("eager", "float16", "cuda:0")
        assert backend == "eager"

    def test_explicit_sdpa(self) -> None:
        backend = resolve_attention_backend("sdpa", "float16", "cuda:0")
        if is_sdpa_available():
            assert backend == "sdpa"
        else:
            assert backend == "eager"


class TestResolveComputePrecision:
    """Tests for compute precision resolution."""

    def test_float16_on_cuda_kept(self) -> None:
        precision = resolve_compute_precision("float16", "cuda:0")
        assert precision == "float16"

    def test_float32_on_cpu_kept(self) -> None:
        precision = resolve_compute_precision("float32", "cpu")
        assert precision == "float32"

    def test_bfloat16_on_unsupported_falls_back(self) -> None:
        # On CPU, bfloat16 is generally supported on modern CPUs
        precision = resolve_compute_precision("bfloat16", "cpu")
        # Should keep bfloat16 on CPU (modern CPUs support it)
        assert precision in ("bfloat16", "float16")


class TestInferenceSettings:
    """Tests for InferenceSettings dataclass."""

    def test_use_fp16_property_true(self) -> None:
        settings = InferenceSettings(
            compute_precision="float16",
            attention_backend="sdpa",
            torch_dtype=torch.float16,
        )
        assert settings.use_fp16 is True

    def test_use_fp16_property_false_fp32(self) -> None:
        settings = InferenceSettings(
            compute_precision="float32",
            attention_backend="sdpa",
            torch_dtype=torch.float32,
        )
        assert settings.use_fp16 is False

    def test_use_fp16_property_false_bf16(self) -> None:
        settings = InferenceSettings(
            compute_precision="bfloat16",
            attention_backend="sdpa",
            torch_dtype=torch.bfloat16,
        )
        assert settings.use_fp16 is False


class TestResolveInferenceSettings:
    """Tests for resolve_inference_settings function."""

    def test_cpu_settings(self) -> None:
        settings = resolve_inference_settings("cpu", compute_precision="float16")
        assert settings.compute_precision == "float16"
        assert settings.torch_dtype == torch.float16
        # CPU should use sdpa or eager (no flash attention)
        assert settings.attention_backend in ("sdpa", "eager")

    def test_fp32_settings(self) -> None:
        settings = resolve_inference_settings("cpu", compute_precision="float32")
        assert settings.compute_precision == "float32"
        assert settings.torch_dtype == torch.float32
        assert settings.attention_backend in ("sdpa", "eager")

    def test_settings_are_frozen(self) -> None:
        settings = resolve_inference_settings("cpu")
        with pytest.raises(AttributeError):
            settings.compute_precision = "float32"  # type: ignore
