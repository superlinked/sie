"""Validate CUDA 13 image dependency contracts without a GPU driver."""

from __future__ import annotations

import argparse
import subprocess
import textwrap

ALLOWED_BUNDLES = ("sglang-cu130", "tensorrt-llm")

COMMON_CHECKS = """
import importlib.metadata
import sie_audio_prep
import sie_sdk
import sie_server
import torch
import transformers

assert torch.__version__.split("+", 1)[0] == "2.11.0", torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda
"""

SGLANG_CHECKS = """
import tvm_ffi
import xgrammar
from sie_server.adapters.sglang import cuda13, gemma, generation

expected = {
    "sglang": "0.5.13",
    "transformers": "5.8.1",
    "kernels": "0.14.1",
    "flashinfer-python": "0.6.12",
    "xgrammar": "0.2.1",
    "apache-tvm-ffi": "0.1.9",
}
for distribution, version in expected.items():
    assert importlib.metadata.version(distribution) == version, distribution
for distribution, upstream in (("sgl-deep-gemm", "0.1.2"), ("sglang-kernel", "0.4.3")):
    actual_upstream, separator, variant = importlib.metadata.version(distribution).partition("+")
    assert (actual_upstream, separator, variant) == (upstream, "+", "cu130"), distribution
assert importlib.metadata.distribution("sglang").files
"""

TENSORRT_LLM_CHECKS = """
import hashlib
import importlib.util
from pathlib import Path

from sie_server.adapters.tensorrt_llm import _server, compat, generation

expected = {
    "tensorrt-llm": "1.3.0rc24",
    "transformers": "5.5.4",
    "flashinfer-python": "0.6.16",
    "cuda-tile": "1.6.0rc8",
    "triton": "3.6.0",
    "nvidia-cutlass-dsl": "4.5.0",
    "nvidia-cuda-tileiras": "13.1.80",
    "nvidia-matmul-heuristics": "0.1.0.27",
    "cuda-pathfinder": "1.7.0",
    "nvidia-nccl-cu13": "2.28.9",
    "flash-attn-4": "4.0.0b11",
    "torch-c-dlpack-ext": "0.1.3",
    "xgrammar": "0.1.32",
    "apache-tvm-ffi": "0.1.6",
    "mpmath": "1.3.0",
}
for distribution, version in expected.items():
    assert importlib.metadata.version(distribution) == version, distribution

engine = importlib.util.find_spec("tensorrt_llm")
assert engine is not None and engine.submodule_search_locations
engine_root = Path(next(iter(engine.submodule_search_locations)))
guarded_sources = {
    engine_root / "_torch/models/checkpoints/hf/weight_loader.py": compat.HF_WEIGHT_LOADER_SHA256,
    engine_root / "_torch/models/modeling_t5.py": compat.MODELING_T5_SHA256,
}
for source, expected_sha256 in guarded_sources.items():
    assert source.is_file(), source
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256, source

from transformers.models.t5 import tokenization_t5
from transformers import tokenization_utils_tokenizers

compat.verify_exact_transformers_5_sources(tokenization_t5, tokenization_utils_tokenizers)
"""


def validation_script(bundle: str) -> str:
    if bundle == "sglang-cu130":
        specific = SGLANG_CHECKS
    elif bundle == "tensorrt-llm":
        specific = TENSORRT_LLM_CHECKS
    else:
        raise ValueError(f"unsupported CUDA 13 bundle: {bundle}")
    return textwrap.dedent(COMMON_CHECKS + specific)


def docker_commands(bundle: str, image_tag: str) -> list[list[str]]:
    script = validation_script(bundle)
    common = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
    ]
    return [
        [*common, "--entrypoint", "python", image_tag, "-c", script],
        [*common, image_tag, "resolve-deps", "--bundle", bundle, "--models-dir", "/app/models"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", choices=ALLOWED_BUNDLES)
    parser.add_argument("image_tag")
    args = parser.parse_args()
    for command in docker_commands(args.bundle, args.image_tag):
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
