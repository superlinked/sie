"""Versioned runtime contract for accelerated training with reference FP32 scoring."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

RECIPE_ID = "laya-fa2-grouped-v1"
REFERENCE_FILES = {
    "bbeh_native.py": "c379f86f55bfccbfcd6c0e613d54adf9f8f155cc1c654a0e30e45998181b9e28",
    "recipe.py": "c500d1278ae7fe5c726f9accd277ff94c8b09b86c78cfbd19c1db1b035ba21e1",
}
VERSIONS = {
    "laya": "0.3.29",
    "peft": "0.18.1",
    "torch": "2.9.1+cu128",
    "transformers": "4.57.6",
    "flash-attn": "2.8.3.post1+cu.12.8.torch.2.9",
    "einops": "0.8.2",
}
LAYA_SOURCES = {
    "laya/train.py": "71c83361dea7f9e4ae0f54d3f5d46e477860ebef875066aac9c9368e3d9fed34",
    "laya/common.py": "f9a67471e635754e3233fa25842ee2a51601755e53f21b2681938021236530e7",
}
EFFECTIVE = {
    "training_attention": "flash_attention_2",
    "training_autocast": "float16",
    "gradient_checkpointing": False,
    "micro_batch": 4,
    "grad_accum": 4,
    "effective_batch_decision_items": 16,
    "grouping": "Stable length sort within each original effective batch; unequal partial tail keeps original order",
    "option_permutations": "Original continuous seeded RNG, drawn in original epoch order before grouping",
    "validation_attention": "sdpa",
    "validation_autocast": False,
    "validation_dtype": "float32",
    "export_attention": "sdpa",
    "selection": "All epochs0..8; minimum development summed binary NLL; earlier epoch breaks exact ties",
    "refit": "Fresh pinned base and seed; train+development for selected epoch count",
    "scheduler": "Original cosine schedule and attempted-window advancement, including GradScaler skips",
    "trajectory": "Grouping changes dropout and floating-point accumulation; this is a distinct recipe",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_reference(directory=None):
    directory = Path(directory) if directory else Path(__file__).parent
    actual = {name: sha(directory / name) for name in REFERENCE_FILES}
    if actual != REFERENCE_FILES:
        raise ValueError("Fast recipe requires the exact frozen reference source")
    return actual


def checked_versions():
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError(f"Fast recipe requires the pinned GPU environment: {actual}")
    return actual


def manifest(plan_sha256, runtime):
    root = Path(__file__).parent
    gpu = root.parent / "gpu"
    return {
        "format": "laya-fast-runtime-recipe-v1",
        "recipe_id": RECIPE_ID,
        "reference_plan_sha256": plan_sha256,
        "reference_sources_sha256": checked_reference(root),
        "implementation_sources_sha256": {
            name: sha(root / name) for name in ("fast_protocol.py", "fast_runtime.py", "fast_train.py")
        },
        "effective_runtime": EFFECTIVE,
        "runtime": runtime,
        "gpu_project_sha256": {name: sha(gpu / name) for name in ("pyproject.toml", "uv.lock")},
        "unchanged": "Inputs, labels, context, loss, LoRA/head dimensions, optimizer hyperparameters and selection rule",
        "final_outcomes_accessed": False,
    }


def runtime_inventory():
    import laya.common
    import laya.train
    import torch
    import transformers.models.modernbert.modeling_modernbert as modernbert

    versions = checked_versions()
    sources = {"laya/train.py": sha(inspect.getfile(laya.train)),
               "laya/common.py": sha(inspect.getfile(laya.common))}
    if sources != LAYA_SOURCES:
        raise ValueError("Installed Laya source differs from the versioned fast implementation contract")
    if not torch.cuda.is_available():
        raise ValueError("The fast recipe requires CUDA")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("TF32 matmul must remain disabled, matching the reference precision")
    return {
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cpu_threads": torch.get_num_threads(),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "installed_source_sha256": dict(sources, **{"transformers/modernbert.py": sha(inspect.getfile(modernbert))}),
    }
