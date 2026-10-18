"""Pinned public model and deterministic, deliberately untrained LoRA fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from safetensors.torch import save_file

from tools.benchmarks.cross_encoder_lora.evidence import digest
from tools.benchmarks.cross_encoder_lora.protocol import MODEL, REVISION, WEIGHTS_SHA256, require


def prepare(output: Path, snapshot: Path | None = None, *, smoke: bool = False) -> dict:
    if snapshot is None:
        snapshot = Path(
            snapshot_download(
                MODEL,
                revision=REVISION,
                token=False,
                allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
            )
        )
    if not smoke:
        require(digest(snapshot / "model.safetensors") == WEIGHTS_SHA256, "pinned model weight hash differs")
    config = json.loads((snapshot / "config.json").read_text())
    require(config["model_type"] == "bert", "the fixture protocol requires BERT")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "snapshot": str(snapshot.resolve()),
        "smoke": smoke,
        "model_files": {p.name: digest(p) for p in sorted(snapshot.iterdir()) if p.is_file()},
        "adapters": {},
    }
    for index, alias in enumerate(("a", "b", "c")):
        seed = 20261016 + index
        directory = output / alias
        directory.mkdir()
        LoraConfig(
            r=4, lora_alpha=8, target_modules=["query", "value"], lora_dropout=0, bias="none", inference_mode=True
        ).save_pretrained(str(directory))
        generator = torch.Generator().manual_seed(seed)
        weights = {}
        width = config["hidden_size"]
        for layer in range(config["num_hidden_layers"]):
            for target in ("query", "value"):
                prefix = f"base_model.model.bert.encoder.layer.{layer}.attention.self.{target}"
                scale = 0.4 if smoke else 0.04
                weights[f"{prefix}.lora_A.weight"] = torch.randn(4, width, generator=generator) * scale
                weights[f"{prefix}.lora_B.weight"] = torch.randn(width, 4, generator=generator) * scale
        save_file(weights, directory / "adapter_model.safetensors")
        manifest["adapters"][alias] = {"seed": seed, "files": {p.name: digest(p) for p in directory.iterdir()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.snapshot)


if __name__ == "__main__":
    main()
