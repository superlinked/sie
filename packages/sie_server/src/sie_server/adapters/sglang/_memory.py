"""Model-sized reservations for SGLang's subprocess-owned CUDA allocations."""

from __future__ import annotations

import json
import logging
import math
import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from sie_server.adapters.sglang import _server

logger = logging.getLogger(__name__)

GIB = 1024**3
RUNTIME_HEADROOM_BYTES = 2 * GIB
TOKEN_ALIGNMENT = 512
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024**2
_FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
_PRECISION_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SGLang automatic memory sizing requires a positive {field}, got {value!r}")
    return value


def _model_directory(model_name_or_path: str, revision: str | None) -> Path:
    path = Path(model_name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import try_to_load_from_cache
    from sie_sdk.cache import get_cache_config

    config_path = try_to_load_from_cache(
        model_name_or_path,
        "config.json",
        revision=revision,
        cache_dir=get_cache_config().local_cache,
    )
    if not isinstance(config_path, str):
        raise ValueError(f"SGLang automatic memory sizing requires the cached model config for {model_name_or_path!r}")
    return Path(config_path).parent


def _weight_bytes(model_dir: Path, element_bytes: int) -> int:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        paths = [model_dir / name for name in sorted(set(index["weight_map"].values()))]
    else:
        paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise ValueError(
            "SGLang automatic memory sizing requires safetensors weights; set mem_fraction_static explicitly"
        )
    total = 0
    for path in paths:
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"Invalid safetensors header size in {path.name}")
            header = json.loads(handle.read(header_size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor["dtype"] in _FLOAT_DTYPES:
                total += math.prod(tensor["shape"]) * element_bytes
            else:
                start, end = tensor["data_offsets"]
                total += end - start
    return _positive_int(total, "weight size")


def _kv_bytes_per_token(config: dict[str, Any], element_bytes: int) -> int:
    text = config.get("text_config", config)
    if (
        config.get("quantization_config")
        or text.get("kv_lora_rank")
        or text.get("linear_num_key_heads")
        or text.get("mamba_d_state")
        or any(kind != "full_attention" for kind in text.get("layer_types", []))
    ):
        raise ValueError("This SGLang cache layout requires an explicit mem_fraction_static")
    layers = _positive_int(text.get("num_hidden_layers"), "num_hidden_layers")
    heads = _positive_int(text.get("num_attention_heads"), "num_attention_heads")
    kv_heads = _positive_int(text.get("num_key_value_heads", heads), "num_key_value_heads")
    head_dim = text.get("head_dim")
    if head_dim is None:
        hidden_size = _positive_int(text.get("hidden_size"), "hidden_size")
        if hidden_size % heads:
            raise ValueError("SGLang automatic memory sizing requires hidden_size divisible by num_attention_heads")
        head_dim = hidden_size // heads
    head_dim = _positive_int(head_dim, "head_dim")
    return 2 * layers * kv_heads * head_dim * element_bytes


def device_memory(device_index: int) -> tuple[int, int]:
    """Return current free and total bytes, including other CUDA processes."""
    import torch

    return torch.cuda.mem_get_info(device_index)


@dataclass
class SGLangMemoryBudget:
    model_name_or_path: str
    revision: str | None
    compute_precision: str
    max_seq_length: int
    kv_budget_tokens: int | None = None
    mem_fraction_static: float | None = None
    tensor_parallel_size: int = 1
    has_auxiliary_weights: bool = False
    reserved_bytes: int = 0

    def __post_init__(self) -> None:
        if self.kv_budget_tokens is not None:
            _positive_int(self.kv_budget_tokens, "kv_budget_tokens")
        fraction = self.mem_fraction_static
        if fraction is not None and (
            isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction <= 1
        ):
            raise ValueError("mem_fraction_static must be a finite number in (0, 1]")

    @property
    def token_capacity(self) -> int:
        budget = self.kv_budget_tokens or _positive_int(self.max_seq_length, "max_seq_length")
        return math.ceil(budget / TOKEN_ALIGNMENT) * TOKEN_ALIGNMENT

    @cached_property
    def static_bytes(self) -> int:
        if self.tensor_parallel_size != 1 or self.has_auxiliary_weights:
            raise ValueError(
                "Tensor parallelism, speculative decoding and LoRA require an explicit mem_fraction_static"
            )
        model_dir = _model_directory(self.model_name_or_path, self.revision)
        config = json.loads((model_dir / "config.json").read_text())
        element_bytes = _PRECISION_BYTES[self.compute_precision]
        weights = _weight_bytes(model_dir, element_bytes)
        kv = _kv_bytes_per_token(config, element_bytes) * (self.token_capacity + TOKEN_ALIGNMENT)
        # The child initializes its CUDA context before measuring free memory.
        # Keep this allowance inside its static fraction; cap the actual KV pool.
        return math.ceil(weights * 1.05) + kv + _server.LOAD_HEADROOM_BYTES

    def required_bytes(self, *, device_type: str, device_total_bytes: int) -> int | None:
        if device_type != "cuda" or device_total_bytes <= 0:
            return None
        if self.mem_fraction_static is not None:
            required = _server.estimate_load_required_memory_bytes(
                device_type=device_type,
                device_total_bytes=device_total_bytes,
                mem_fraction_static=self.mem_fraction_static,
            )
            self.reserved_bytes = (required or 0) * self.tensor_parallel_size
            return required
        required = self.static_bytes + RUNTIME_HEADROOM_BYTES
        self.reserved_bytes = required
        return required

    def launch_args(self, device_index: int) -> list[str]:
        if self.mem_fraction_static is not None:
            return ["--mem-fraction-static", str(self.mem_fraction_static)]
        free, total = device_memory(device_index)
        required = self.required_bytes(device_type="cuda", device_total_bytes=total)
        if required is None or free < required:
            raise RuntimeError(f"CUDA out of memory: SGLang needs {required} bytes free, but only {free} are available")
        # SGLang 0.5.10.post1 profiles KV from post-load free minus
        # pre-load free * (1 - fraction), not a fraction of total VRAM.
        fraction = self.static_bytes / free
        logger.info(
            "SGLang memory reservation for %s: %.2f GiB, KV capacity %d tokens, mem_fraction_static=%.6f",
            self.model_name_or_path,
            required / GIB,
            self.token_capacity,
            fraction,
        )
        return ["--mem-fraction-static", str(fraction), "--max-total-tokens", str(self.token_capacity)]
