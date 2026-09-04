"""Device detection utilities for GPU/accelerator selection."""

from __future__ import annotations

import platform
import subprocess


def detect_gpu() -> str | None:
    """Detect available GPU type.

    Returns:
        "cuda" if NVIDIA GPU is available (nvidia-smi works),
        "mps" if Apple Silicon is detected,
        None if no GPU is available.
    """
    system = platform.system()

    # Check for Apple Silicon (MPS)
    if system == "Darwin" and platform.machine() == "arm64":
        return "mps"

    # Check for NVIDIA GPU (CUDA)
    if system in ("Linux", "Windows"):
        try:
            result = subprocess.run(
                ["nvidia-smi"],  # noqa: S607 — intentional partial path
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return "cuda"
        except FileNotFoundError:
            pass

    return None


def default_device() -> str:
    """Get the default device string for PyTorch.

    Returns:
        "cuda:0" if NVIDIA GPU is available,
        "mps:0" if Apple Silicon is detected,
        "cpu" otherwise.
    """
    gpu = detect_gpu()
    if gpu == "cuda":
        return "cuda:0"
    if gpu == "mps":
        return "mps:0"
    return "cpu"
