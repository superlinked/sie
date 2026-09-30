r"""Generate parity goldens for the TopK-Embed-V1 adapter from TopK's own pipeline.

Runs the checkpoint's sentence-transformers stack (``MultiVectorEncoder`` with its
remote code) on fixed queries, text documents and synthetic pages, without any SIE
code, and writes per-input token counts, token ids, per-token projections onto
fixed random directions, and the MaxSim score matrices to
``tests/adapters/goldens/topk_embed/<org>__<name>.json``.

The pipeline's backbone (``hf_backbone.py``) needs CUDA-only kernels:
flash-linear-attention's ``causal_conv1d`` and ``chunk_gated_delta_rule`` over
packed sequences, and ``torch.compile``'d flex attention. On CUDA the script runs
it unchanged (install ``flash-linear-attention``). With ``--device cpu`` it swaps
those three for reference implementations with the same math: a depthwise causal
conv per packed segment, transformers' torch ``chunk_gated_delta_rule`` per
segment, and eager flex attention with the same document mask. Everything else,
preprocessing included, is TopK's code.

Run with the checkpoint's own requirements (``requirements.txt`` in the model
repo; transformers 5.9 and sentence-transformers 6.0):

    uv run --no-project --python 3.12 \\
        --with torch==2.9.1 --with torchvision==0.24.1 --with transformers==5.9.0 \\
        --with sentence-transformers==6.0.1 --with safetensors --with pillow \\
        python packages/sie_server/scripts/generate_topk_embed_goldens.py --device cpu
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import itertools
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODELS = {
    "topk-io/topk-embed-v1-xsmall": "d09d8a7a8cdd6c287f792b3c4d7b41233d46e66a",
    "topk-io/topk-embed-v1-small": "33b15d544d74f29d04cdb97adeb9fd0e52da5fb7",
}
QUERIES = [
    "how does photosynthesis work?",
    "  what was the Q3 revenue  ",
    "",
]
DOCUMENTS = [
    "Plants convert light energy into chemical energy stored as glucose.",
    "",
    "!!! ... ??? --- (( )) <|im_end|> <|endoftext|>",
    " ".join(f"Section {i}: revenue grew in region {i % 5}, driven by renewals (table {i})." for i in range(12)),
]
# Synthetic pages: dark "text line" bars on a light background, from a fixed seed.
PAGES = [
    {"name": "letter", "width": 850, "height": 1100, "seed": 1},
    {"name": "portrait", "width": 480, "height": 640, "seed": 2},
    {"name": "small", "width": 300, "height": 200, "seed": 3},
]
# Per-token vectors are stored as projections onto these directions (unit, seeded).
N_DIRECTIONS = 4
GOLDENS_DIR = Path(__file__).resolve().parents[1] / "tests" / "adapters" / "goldens" / "topk_embed"


def render_page(spec: dict[str, Any]) -> Image.Image:
    rng = np.random.default_rng(spec["seed"])
    height, width = spec["height"], spec["width"]
    page = np.full((height, width, 3), 245, dtype=np.uint8)
    line_height = max(6, height // 40)
    for top in range(height // 12, height - height // 12, line_height * 2):
        length = int(width * rng.uniform(0.3, 0.8))
        shade = int(rng.integers(10, 60))
        page[top : top + line_height, width // 10 : width // 10 + length] = shade
    return Image.fromarray(page)


def directions(dim: int) -> np.ndarray:
    matrix = np.random.default_rng(0).standard_normal((N_DIRECTIONS, dim))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def install_cpu_kernels() -> None:
    """Provide the flash-linear-attention symbols hf_backbone.py imports, as torch code."""

    def causal_conv1d(x, weight, bias=None, activation=None, cu_seqlens=None, **_):
        bounds = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        outs = []
        for start, end in itertools.pairwise(bounds):
            segment = F.pad(x[:, start:end].transpose(1, 2), (weight.shape[-1] - 1, 0))
            y = F.conv1d(segment, weight.unsqueeze(1), bias, groups=x.shape[-1])
            outs.append((F.silu(y) if activation in ("silu", "swish") else y).transpose(1, 2))
        return torch.cat(outs, dim=1), None

    def module(name: str, *, package: bool = False) -> types.ModuleType:
        stub = types.ModuleType(name)
        # transformers probes installed packages with importlib.util.find_spec.
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
        return stub

    fla = module("fla", package=True)
    fla_utils = module("fla.utils")
    fla_utils.FLA_DISABLE_TENSOR_CACHE = False  # ty: ignore[unresolved-attribute]
    fla_modules = module("fla.modules", package=True)
    fla_convolution = module("fla.modules.convolution")
    fla_convolution.causal_conv1d = causal_conv1d  # ty: ignore[unresolved-attribute]
    fla.utils = fla_utils  # ty: ignore[unresolved-attribute]
    fla.modules = fla_modules  # ty: ignore[unresolved-attribute]
    fla_modules.convolution = fla_convolution  # ty: ignore[unresolved-attribute]
    sys.modules.update(
        {"fla": fla, "fla.utils": fla_utils, "fla.modules": fla_modules, "fla.modules.convolution": fla_convolution}
    )


def patch_cpu_model(model: Any) -> None:
    """Eager flex attention, and the torch delta rule run once per packed segment."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    backbone = next(module for name, module in sys.modules.items() if name.endswith("hf_backbone"))
    backbone.compiled_flex_attention = flex_attention  # ty: ignore[unresolved-attribute]
    backbone.compiled_create_block_mask = create_block_mask  # ty: ignore[unresolved-attribute]

    def segmented(rule):
        def run(query, key, value, *, g, beta, cu_seqlens=None, **kwargs):
            bounds = [0, query.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
            outs = [
                rule(query[:, s:e], key[:, s:e], value[:, s:e], g=g[:, s:e], beta=beta[:, s:e], **kwargs)[0]
                for s, e in itertools.pairwise(bounds)
            ]
            return torch.cat(outs, dim=1), None

        return run

    for layer in model[0].auto_model.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = segmented(layer.linear_attn.chunk_gated_delta_rule)


def dumps(golden: dict[str, Any]) -> str:
    """Indented JSON with every list of numbers on one line."""
    text = json.dumps(golden, indent=1)
    return re.sub(r"\[([^\[\]{}\"]*)\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text) + "\n"


def maxsim(queries: list[np.ndarray], documents: list[np.ndarray]) -> list[list[float]]:
    return [[float((q @ d.T).max(axis=1).sum()) for d in documents] for q in queries]


def generate(model_id: str, revision: str, device: str) -> dict[str, Any]:
    from sentence_transformers import MultiVectorEncoder  # ty: ignore[unresolved-import]

    model = MultiVectorEncoder(model_id, revision=revision, trust_remote_code=True, device=device)
    if device == "cpu":
        patch_cpu_model(model)
    module = model[0]
    pages = [render_page(spec) for spec in PAGES]
    with torch.inference_mode():
        queries = [t.float().cpu().numpy() for t in model.encode_query(QUERIES)]
        documents = [t.float().cpu().numpy() for t in model.encode_document(DOCUMENTS)]
        images = [model.encode_document([page])[0].float().cpu().numpy() for page in pages]
    basis = directions(queries[0].shape[1])

    def entry(vectors: np.ndarray, **extra: Any) -> dict[str, Any]:
        # One row per direction, one value per token.
        return {"tokens": len(vectors), "projections": np.round(basis @ vectors.T, 5).tolist(), **extra}

    query_ids = module.preprocess(QUERIES, task="query")
    doc_ids = module.preprocess(DOCUMENTS, task="document")
    return {
        "model": model_id,
        "revision": revision,
        "generated_with": {
            "pipeline": "sentence_transformers.MultiVectorEncoder + the checkpoint's remote code",
            "device": device,
            "kernels": "reference torch/eager stand-ins (see script)" if device == "cpu" else "native",
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "directions": {"seed": 0, "count": N_DIRECTIONS, "distribution": "standard normal, unit rows"},
        },
        "queries": [
            entry(v, text=t, input_ids=row[mask].tolist())
            for v, t, row, mask in zip(
                queries, QUERIES, query_ids["input_ids"], query_ids["attention_mask"], strict=True
            )
        ],
        "documents": [
            entry(v, text=t, input_ids=row[mask].tolist())
            for v, t, row, mask in zip(
                documents, DOCUMENTS, doc_ids["input_ids"], doc_ids["attention_mask"], strict=True
            )
        ],
        "pages": [entry(v, **spec) for v, spec in zip(images, PAGES, strict=True)],
        "text_scores": maxsim(queries, documents),
        "page_scores": maxsim(queries, images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parity goldens for the TopK-Embed-V1 adapter.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", choices=sorted(MODELS), action="append")
    args = parser.parse_args()
    if args.device == "cpu":
        install_cpu_kernels()
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in args.model or list(MODELS):
        golden = generate(model_id, MODELS[model_id], args.device)
        path = GOLDENS_DIR / f"{model_id.replace('/', '__')}.json"
        path.write_text(dumps(golden), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
