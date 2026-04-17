"""Inference optimization utilities.

Handles compute precision and attention backend configuration per DESIGN.md Section 6.3-6.4.

Compute precision pipeline:
- Weights on disk (FP32/BF16) → Load/Cast to GPU → Compute (FP16/BF16/FP32) → Output

Attention backend selection:
- Flash Attention 2: Ampere+ GPUs (A100, RTX 30xx+), AMD MI210/250/300 - 2-4x speedup
- SDPA: All GPUs (Volta+) - 1.5-2x speedup, fallback when FA2 unavailable
- Eager: Baseline, no optimizations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Type aliases matching engine config
AttentionBackend = Literal["auto", "flash_attention_2", "sdpa", "eager"]
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Ampere is the minimum architecture for Flash Attention 2 and BF16 support
_MIN_AMPERE_COMPUTE_CAPABILITY = 8


@dataclass(frozen=True)
class InferenceSettings:
    """Resolved inference settings for a model.

    These settings are determined by combining engine defaults with model overrides.
    """

    compute_precision: ComputePrecision
    attention_backend: AttentionBackend
    torch_dtype: torch.dtype

    @property
    def use_fp16(self) -> bool:
        """Whether FP16 is being used (for adapters that need a boolean flag)."""
        return self.compute_precision == "float16"


def get_torch_dtype(precision: ComputePrecision) -> torch.dtype:
    """Convert precision string to torch dtype.

    Args:
        precision: Precision string from config.

    Returns:
        Corresponding torch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[precision]


def is_flash_attention_available(device: str | None = None) -> bool:
    """Check if Flash Attention 2 is available.

    Flash Attention 2 requires:
    - CUDA device with Ampere or newer architecture (compute capability >= 8.0)
    - flash-attn package installed
    - Transformers with FA2 support

    Args:
        device: Target device string (e.g. ``"cuda:0"``).  When provided the
            capability check runs against that specific GPU rather than
            ``torch.cuda.current_device()``, which matters on multi-GPU hosts.

    Returns:
        True if Flash Attention 2 can be used.
    """
    if not torch.cuda.is_available():
        return False

    # Check CUDA compute capability (Ampere is 8.0+)
    try:
        if device is not None and ":" in device:
            device_idx = int(device.split(":")[1])
        else:
            device_idx = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_idx)
        if capability[0] < _MIN_AMPERE_COMPUTE_CAPABILITY:
            logger.debug(
                "Flash Attention 2 requires Ampere+ GPU (compute capability 8.0+), current device has %d.%d",
                capability[0],
                capability[1],
            )
            return False
    except (RuntimeError, AssertionError, ValueError):
        logger.debug("Could not determine CUDA compute capability", exc_info=True)
        return False

    # Check if flash-attn package is available
    import importlib.util

    if importlib.util.find_spec("flash_attn") is None:
        logger.debug("flash-attn package not installed, Flash Attention 2 unavailable")
        return False
    return True


def is_sdpa_available() -> bool:
    """Check if Scaled Dot-Product Attention (SDPA) is available.

    SDPA is available in PyTorch 2.0+ and works on all GPUs.

    Returns:
        True if SDPA can be used.
    """
    # SDPA is available in PyTorch 2.0+
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    return torch_version >= (2, 0)


def is_bfloat16_supported(device: str) -> bool:
    """Check if bfloat16 is supported on the device.

    BF16 is supported on:
    - Ampere+ GPUs (compute capability >= 8.0)
    - CPU (with appropriate support)
    - Apple Silicon (MPS)

    Args:
        device: Device string (e.g., "cuda:0", "cpu", "mps").

    Returns:
        True if bfloat16 can be used on this device.
    """
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return False
        try:
            device_idx = int(device.rsplit(":", maxsplit=1)[-1]) if ":" in device else 0
            capability = torch.cuda.get_device_capability(device_idx)
            return capability[0] >= _MIN_AMPERE_COMPUTE_CAPABILITY
        except (RuntimeError, ValueError):
            return False

    # MPS (Apple Silicon) and CPU support bfloat16
    return device in ("mps", "cpu")


def _get_sdpa_or_eager() -> AttentionBackend:
    """Return SDPA if available, else eager."""
    return "sdpa" if is_sdpa_available() else "eager"


def resolve_attention_backend(
    requested: AttentionBackend,
    compute_precision: ComputePrecision,
    device: str,
) -> AttentionBackend:
    """Resolve the attention backend to use.

    Handles "auto" selection and validates compatibility with precision and device.

    Per DESIGN.md: Flash Attention 2 requires FP16 or BF16 and CUDA device.

    Args:
        requested: Requested backend from config.
        compute_precision: Compute precision being used.
        device: Target device string (e.g., "cuda:0", "cpu").

    Returns:
        Resolved attention backend to use.
    """
    is_cuda = device.startswith("cuda")

    # Flash Attention 2 requires CUDA
    if requested == "flash_attention_2" and not is_cuda:
        logger.warning("Flash Attention 2 requires CUDA, falling back to SDPA for %s", device)
        return _get_sdpa_or_eager()

    # FP32 cannot use Flash Attention 2
    if compute_precision == "float32" and requested == "flash_attention_2":
        logger.warning("Flash Attention 2 requires FP16 or BF16, falling back to SDPA for FP32")
        return _get_sdpa_or_eager()

    if requested == "auto":
        return _resolve_auto_backend(compute_precision, is_cuda)

    # Validate explicit requests
    return _validate_explicit_backend(requested)


def _resolve_auto_backend(compute_precision: ComputePrecision, is_cuda: bool) -> AttentionBackend:
    """Auto-select the best available attention backend."""
    if is_cuda and compute_precision != "float32" and is_flash_attention_available():
        logger.info("Auto-selected Flash Attention 2")
        return "flash_attention_2"
    if is_sdpa_available():
        logger.info("Auto-selected SDPA (Scaled Dot-Product Attention)")
        return "sdpa"
    logger.info("Using eager attention (no optimizations available)")
    return "eager"


def _validate_explicit_backend(requested: AttentionBackend) -> AttentionBackend:
    """Validate an explicitly requested backend."""
    if requested == "flash_attention_2" and not is_flash_attention_available():
        logger.warning("Flash Attention 2 requested but not available, falling back to SDPA")
        return _get_sdpa_or_eager()

    if requested == "sdpa" and not is_sdpa_available():
        logger.warning("SDPA requested but not available, falling back to eager")
        return "eager"

    return requested


def resolve_compute_precision(
    requested: ComputePrecision,
    device: str,
) -> ComputePrecision:
    """Resolve compute precision with device compatibility.

    Args:
        requested: Requested precision from config.
        device: Target device string.

    Returns:
        Validated compute precision that works on the device.
    """
    # BF16 requires specific hardware support
    if requested == "bfloat16" and not is_bfloat16_supported(device):
        logger.warning(
            "BFloat16 requested but not supported on %s, falling back to FP16",
            device,
        )
        return "float16"

    # CPU typically uses FP32 for best compatibility
    if device == "cpu" and requested in ("float16", "bfloat16"):
        # FP16 on CPU is very slow, but we respect explicit config
        logger.debug("Using %s on CPU - consider FP32 for better performance", requested)

    return requested


def resolve_inference_settings(
    device: str,
    *,
    compute_precision: ComputePrecision = "float16",
    attention_backend: AttentionBackend = "auto",
) -> InferenceSettings:
    """Resolve complete inference settings for a device.

    Combines precision and attention backend resolution.

    Args:
        device: Target device string (e.g., "cuda:0", "cpu").
        compute_precision: Requested compute precision.
        attention_backend: Requested attention backend.

    Returns:
        InferenceSettings with validated, compatible settings.
    """
    # Resolve precision first (affects attention backend selection)
    resolved_precision = resolve_compute_precision(compute_precision, device)

    # Resolve attention backend with precision and device context
    resolved_backend = resolve_attention_backend(attention_backend, resolved_precision, device)

    # Get torch dtype
    torch_dtype = get_torch_dtype(resolved_precision)

    settings = InferenceSettings(
        compute_precision=resolved_precision,
        attention_backend=resolved_backend,
        torch_dtype=torch_dtype,
    )

    logger.info(
        "Inference settings for %s: precision=%s, attention=%s",
        device,
        resolved_precision,
        resolved_backend,
    )

    return settings
