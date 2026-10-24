"""Check the locked CUDA environment with a synthetic ModernBERT LoRA step."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import tomllib
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    root = Path(__file__).resolve().parent
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Linux x86_64 and Python 3.12 with the gpu/ project")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this check with uv run --project gpu --frozen")
    dependencies = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    expected = dict(item.split("==", 1) for item in dependencies)
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"Installed versions differ from gpu/pyproject.toml: {versions}")
    locations = {name: str(Path(importlib.metadata.distribution(name).locate_file("")).resolve()) for name in expected}
    if any(not Path(path).is_relative_to(Path(sys.prefix).resolve()) for path in locations.values()):
        raise RuntimeError("Training dependencies must be installed in this project's virtual environment")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "virtual_environment": sys.prefix,
        "versions": versions,
        "package_locations": locations,
        "files_sha256": {name: sha(root / name) for name in ("pyproject.toml", "uv.lock", "smoke.py")},
    }


def check():
    receipt = environment()
    # These optional imports deliberately exercise the complete training import chain.
    import flash_attn
    import laya.train
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import ModernBertConfig, ModernBertModel

    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("A working NVIDIA driver and the locked CUDA 12.8 Torch build are required")
    if not torch.compiled_with_cxx11_abi():
        raise RuntimeError("The pinned FlashAttention wheel requires Torch's C++11 ABI")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("FlashAttention 2 requires an Ampere or newer NVIDIA GPU")

    torch.manual_seed(89)
    config = ModernBertConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8192,
        local_attention=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        reference_compile=False, attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"
    model = get_peft_model(
        ModernBertModel(config).to("cuda"),
        LoraConfig(r=4, lora_alpha=8, target_modules=["Wqkv"], lora_dropout=0.0, bias="none"),
    ).train()
    head = torch.nn.Linear(128, 2, device="cuda")
    parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    before = head.weight.detach().clone()
    ids = torch.randint(3, 128, (2, 256), device="cuda")
    mask = torch.ones_like(ids)
    mask[1, 192:] = 0
    ids[mask == 0] = 0
    target = torch.tensor([0, 1], device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        states = model(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        logits = head(pooled)
        loss = torch.nn.functional.cross_entropy(logits.float(), target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise RuntimeError("The synthetic forward pass produced a nonfinite value")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError("The synthetic backward pass produced a nonfinite gradient")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    steps = {int(state["step"]) for state in optimizer.state.values()}
    if steps != {1} or torch.equal(before, head.weight) or any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError("The synthetic optimizer did not complete one finite parameter update")
    unused = [name for name in sys.modules if name == "flash_attn.utils.distributed" or name.startswith("flash_attn.models")]
    if unused:
        raise RuntimeError(f"Unexpected FlashAttention checkpoint helper import: {unused}")
    receipt.update({
        "format": "router-gpu-install-smoke-v1", "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda_runtime": torch.version.cuda, "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "imported_sources_sha256": {"flash_attn": sha(flash_attn.__file__), "laya.train": sha(laya.train.__file__)},
        "attention": config._attn_implementation, "parameter_dtype": str(head.weight.dtype),
        "autocast": "float16", "actual_optimizer_steps": 1, "loss": float(loss.detach()),
        "sequence_lengths": [256, 192], "step_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "Synthetic import/forward/backward/update check; no checkpoint, API, benchmark data or quality evaluation",
        "unused_checkpoint_helpers_imported": unused,
    })
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new receipt path")
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "actual_optimizer_steps": 1}))


if __name__ == "__main__":
    main()
