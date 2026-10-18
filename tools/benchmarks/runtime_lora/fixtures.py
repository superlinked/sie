"""Deterministic, deliberately untrained LoRAs for runtime correctness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file

MODELS = {
    "generation": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "embedding": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
}
ALIASES = ("pin", "a", "b", "c", "d")
RANK = 4
SEED = 20261016


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_adapter(directory: Path, config: dict[str, Any], model: str, revision: str, seed: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = config["hidden_size"]
    head = config["head_dim"]
    widths = {"q_proj": config["num_attention_heads"] * head, "v_proj": config["num_key_value_heads"] * head}
    tensors = {}
    for layer in range(config["num_hidden_layers"]):
        for module, width in widths.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(RANK, hidden, generator=generator) * 0.04
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(width, RANK, generator=generator) * 0.04
    adapter_config = {
        "base_model_name_or_path": model,
        "revision": revision,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": RANK,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(widths),
        "inference_mode": True,
    }
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    return {
        "seed": seed,
        "rank": RANK,
        "target_modules": list(widths),
        "tensor_count": len(tensors),
        "files": {name: digest(directory / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
    }


def prepare(output: Path, *, download_weights: bool = False, cache_dir: str | None = None) -> dict[str, Any]:
    manifest = {}
    for kind, (model, revision) in MODELS.items():
        config_path = Path(hf_hub_download(model, "config.json", revision=revision, cache_dir=cache_dir))
        config = json.loads(config_path.read_text())
        if config["model_type"] != "qwen3":
            raise ValueError("This fixture generator requires the pinned Qwen3 architecture")
        if download_weights:
            snapshot_download(
                model,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json", "*.txt"],
            )
        manifest[kind] = {
            "model": model,
            "revision": revision,
            "config_sha256": digest(config_path),
            "shape": {
                key: config[key]
                for key in (
                    "hidden_size",
                    "head_dim",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "num_hidden_layers",
                )
            },
            "adapters": {
                alias: write_adapter(output / kind / alias, config, model, revision, SEED + index)
                for index, alias in enumerate(ALIASES)
            },
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--download-weights", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.output.resolve(), download_weights=args.download_weights, cache_dir=args.cache_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
