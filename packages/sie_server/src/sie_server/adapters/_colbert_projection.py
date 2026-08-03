"""Shared loading for standalone ColBERT projection weights.

Stanford-format ColBERT checkpoints can store a trained projection head at the
repository root as ``linear.weight`` without registering that layer on the
Hugging Face backbone returned by ``AutoModel``.  This helper keeps that
checkpoint-layout handling consistent across the generic and rotary adapters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

logger = logging.getLogger(__name__)


class ColBERTProjectionLoadError(RuntimeError):
    """Raised when a required trained ColBERT projection cannot be loaded."""


def _failure(message: str, *, required: bool) -> None:
    if required:
        raise ColBERTProjectionLoadError(message)
    logger.debug(message)


def _root_safetensors_path(model_name_or_path: str, *, revision: str | None) -> str:
    local = Path(model_name_or_path) / "model.safetensors"
    if local.is_file():
        return str(local)
    return hf_hub_download(model_name_or_path, "model.safetensors", revision=revision)


def load_standalone_colbert_projection(
    model_name_or_path: str,
    *,
    revision: str | None,
    device: str | torch.device | None,
    dtype: torch.dtype,
    expected_in_features: int | None = None,
    expected_out_features: int | None = None,
    allow_bias: bool = True,
    required: bool = False,
) -> torch.nn.Linear | None:
    """Load a root ``linear.weight`` projection from one exact model revision.

    Optional mode preserves the generic ColBERT adapter's historical contract:
    a checkpoint without this standalone layout returns ``None`` so its other
    projection probes or intentional Matryoshka fallback can run.  Required
    mode fails closed for adapters whose served model contract depends on this
    trained head.
    """
    try:
        weights_path = _root_safetensors_path(model_name_or_path, revision=revision)
    except Exception as exc:  # noqa: BLE001 - Hub/cache failures share the required/optional policy
        _failure(
            f"Unable to resolve root model.safetensors for {model_name_or_path} at revision {revision}: {exc}",
            required=required,
        )
        return None

    try:
        with safe_open(weights_path, framework="pt") as checkpoint:
            keys = set(checkpoint.keys())
            if "linear.weight" not in keys:
                _failure(
                    f"Checkpoint {model_name_or_path} at revision {revision} has no standalone linear.weight",
                    required=required,
                )
                return None

            weight = checkpoint.get_tensor("linear.weight")
            if weight.ndim != 2:
                _failure(
                    f"Checkpoint {model_name_or_path} linear.weight must be 2D, got shape {tuple(weight.shape)}",
                    required=required,
                )
                return None

            out_features, in_features = weight.shape
            if expected_in_features is not None and in_features != expected_in_features:
                _failure(
                    f"Checkpoint {model_name_or_path} linear.weight in_features={in_features} "
                    f"does not match model hidden_size={expected_in_features}",
                    required=required,
                )
                return None
            if expected_out_features is not None and out_features != expected_out_features:
                _failure(
                    f"Checkpoint {model_name_or_path} linear.weight out_features={out_features} "
                    f"does not match configured token_dim={expected_out_features}",
                    required=required,
                )
                return None

            has_bias = "linear.bias" in keys
            if has_bias and not allow_bias:
                _failure(
                    f"Checkpoint {model_name_or_path} has unsupported linear.bias; a bias-free projection is required",
                    required=required,
                )
                return None

            bias = checkpoint.get_tensor("linear.bias") if has_bias else None
            if bias is not None and (bias.ndim != 1 or bias.shape[0] != out_features):
                _failure(
                    f"Checkpoint {model_name_or_path} linear.bias shape {tuple(bias.shape)} "
                    f"does not match out_features={out_features}",
                    required=required,
                )
                return None
    except ColBERTProjectionLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - corrupt/partial checkpoints follow required/optional policy
        _failure(
            f"Unable to read standalone projection for {model_name_or_path} at revision {revision}: {exc}",
            required=required,
        )
        return None

    projection = torch.nn.Linear(in_features, out_features, bias=has_bias)
    with torch.no_grad():
        projection.weight.copy_(weight)
        if bias is not None:
            assert projection.bias is not None
            projection.bias.copy_(bias)
    projection = projection.to(device=device, dtype=dtype)
    projection.eval()
    logger.info(
        "Loaded standalone ColBERT projection for %s at revision %s: %d -> %d",
        model_name_or_path,
        revision,
        in_features,
        out_features,
    )
    return projection
