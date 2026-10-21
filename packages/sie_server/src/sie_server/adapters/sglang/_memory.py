"""Model-sized reservations for SGLang's subprocess-owned CUDA allocations.

SIE sizes an engine from the pinned checkpoint's safetensors headers and
``config.json``; no weights are loaded onto a GPU. The reservation covers what
the engine allocates inside its static memory fraction: the resident weights,
the KV pool for the profile's token budget, the recurrent-state pool of hybrid
linear-attention models, LoRA buffers, and a startup allowance. Runtime
headroom covers what the engine allocates after it, and online quantization
adds a separate, higher figure for the free memory loading needs. SIE pins
the engine pools the reservation assumes. A configuration whose allocations
cannot be derived from that metadata raises and must pin
``mem_fraction_static`` instead.

The formulas mirror the pinned engines (SGLang 0.5.10.post1 and 0.5.20); the
comments name the engine code each one follows.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from sie_server.adapters.sglang import _server

logger = logging.getLogger(__name__)

GIB = 1024**3
RUNTIME_HEADROOM_BYTES = 2 * GIB
TOKEN_ALIGNMENT = 512
WEIGHT_ALLOWANCE = 1.05
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024**2
_FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
_PRECISION_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
_STATE_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
# ``--kv-cache-dtype`` values with a plain element size. ``auto`` is resolved
# separately; block-scaled formats (fp4, mxfp8) carry scale buffers and stay
# on an explicit fraction.
_KV_CACHE_DTYPE_BYTES = {"bf16": 2, "bfloat16": 2, "fp8_e4m3": 1, "fp8_e5m2": 1}
# Checkpoint quantization whose serialized tensors are loaded unchanged:
# blockwise or per-tensor FP8 weights keep their FP8 storage, and SGLang holds
# their scales in float32.
_SERIALIZED_QUANTIZATION = {"fp8"}
# Online quantization of an unquantized checkpoint. The engine loads every
# weight at the serving precision, then quantizes decoder linear weights to
# one byte, so loading peaks at the unquantized size.
_ONLINE_QUANTIZATION = {"fp8"}
_SCALE_SUFFIXES = ("_scale", "_scale_inv")
_TOWER_MARKERS = ("visual", "vision", "audio")
# Mamba state slots per running request for each radix-cache strategy, as the
# engines cap ``max_running_requests`` at ``max_mamba_cache_size // ratio``.
_MAMBA_STRATEGY_RATIO = {"no_buffer": 3, "extra_buffer_lazy": 4}
# What ``auto`` resolves to, keyed by the strategy flag each engine spells:
# 0.5.10 always picks no_buffer. 0.5.20 picks extra_buffer when the overlap
# scheduler or paging is on and no_buffer otherwise; extra_buffer is the
# larger reservation, so SIE assumes it.
_MAMBA_AUTO_STRATEGY = {"--mamba-scheduler-strategy": "no_buffer", "--mamba-radix-cache-strategy": "extra_buffer"}
# The engines' LoRA module normalization (get_normalized_target_modules).
_LORA_MODULES = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "qkv_proj": "qkv_proj",
    "o_proj": "o_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "gate_up_proj": "gate_up_proj",
    "down_proj": "down_proj",
}
_PROJECTION = re.compile(
    r"\.layers\.(?P<layer>\d+)\.(?:[\w.]+\.)?"
    r"(?P<module>q_proj|k_proj|v_proj|qkv_proj|o_proj|gate_proj|up_proj|gate_up_proj|down_proj)\.weight$"
)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SGLang automatic memory sizing requires a positive {field}, got {value!r}")
    return value


def flag_value(args: Sequence[str], flag: str) -> str | None:
    """Return the last value given for ``flag`` in either argv spelling."""
    value = None
    for index, arg in enumerate(args):
        name, sep, inline = arg.partition("=")
        if name != flag:
            continue
        if sep:
            value = inline
        elif index + 1 < len(args):
            value = args[index + 1]
    return value


def _flag_int(args: Sequence[str], flag: str) -> int | None:
    value = flag_value(args, flag)
    if value is None:
        return None
    try:
        return _positive_int(int(value), flag)
    except ValueError as error:
        raise ValueError(f"SGLang automatic memory sizing requires a positive integer {flag}, got {value!r}") from error


def _has_flag(args: Sequence[str], flag: str) -> bool:
    return any(arg.partition("=")[0] == flag for arg in args)


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


def _tensors(model_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
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
    for path in paths:
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"Invalid safetensors header size in {path.name}")
            header = json.loads(handle.read(header_size))
        for name, tensor in header.items():
            if name != "__metadata__":
                yield name, tensor


def _quantizes_online(name: str, shape: list[int]) -> bool:
    """Whether online FP8 stores this tensor in one byte per element.

    Decoder linear weights are quantized. Embeddings, the output head, norms
    and convolutions keep the serving precision, and so do vision and audio
    towers: some engines leave them unquantized, so counting them in full
    keeps the estimate conservative.
    """
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and ".layers." in name
        and not any(marker in name for marker in _TOWER_MARKERS)
    )


def _weight_bytes(
    model_dir: Path,
    element_bytes: int,
    *,
    serialized_quantization: bool = False,
    online_quantization: bool = False,
    include_mtp: bool = False,
) -> tuple[int, int]:
    """Return resident and peak-at-load weight bytes.

    Floating tensors are counted at the serving precision; quantized tensors
    at their stored size. The target model skips multi-token-prediction
    weights (``mtp.*``) unless a speculative draft loads them.
    """
    resident = peak = 0
    for name, tensor in _tensors(model_dir):
        if not include_mtp and name.startswith("mtp."):
            continue
        start, end = tensor["data_offsets"]
        stored = end - start
        numel = math.prod(tensor["shape"])
        if tensor["dtype"] not in _FLOAT_DTYPES:
            resident += stored
            peak += stored
        elif serialized_quantization and name.endswith(_SCALE_SUFFIXES):
            resident += numel * 4
            peak += numel * 4
        else:
            peak += numel * element_bytes
            resident += (
                numel if online_quantization and _quantizes_online(name, tensor["shape"]) else numel * element_bytes
            )
    return _positive_int(resident, "weight size"), peak


@dataclass(frozen=True)
class _AttentionLayout:
    full: int
    sliding: int
    linear: int


def _attention_layout(text: dict[str, Any]) -> _AttentionLayout:
    layers = _positive_int(text.get("num_hidden_layers"), "num_hidden_layers")
    kinds = text.get("layer_types")
    if kinds is None:
        interval = text.get("full_attention_interval")
        if text.get("linear_num_key_heads") and isinstance(interval, int) and interval > 0:
            kinds = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
        else:
            kinds = ["full_attention"] * layers
    if len(kinds) != layers:
        raise ValueError("SGLang automatic memory sizing requires one layer_types entry per hidden layer")
    unknown = set(kinds) - {"full_attention", "sliding_attention", "linear_attention"}
    if unknown:
        raise ValueError(
            f"The SGLang cache layout for {sorted(unknown)} layers requires an explicit mem_fraction_static"
        )
    return _AttentionLayout(
        full=kinds.count("full_attention"),
        sliding=kinds.count("sliding_attention"),
        linear=kinds.count("linear_attention"),
    )


def _quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    return config.get("quantization_config") or text.get("quantization_config") or {}


def _kv_element_bytes(config: dict[str, Any], args: Sequence[str], element_bytes: int) -> int:
    requested = flag_value(args, "--kv-cache-dtype") or "auto"
    if requested == "auto":
        # Both engines switch ``auto`` to FP8 only for a checkpoint that
        # declares an FP8 KV-cache quantization algorithm.
        algo = _quantization_config(config).get("kv_cache_quant_algo")
        return 1 if isinstance(algo, str) and algo.upper() == "FP8" else element_bytes
    if requested not in _KV_CACHE_DTYPE_BYTES:
        raise ValueError(f"--kv-cache-dtype {requested} requires an explicit mem_fraction_static")
    return _KV_CACHE_DTYPE_BYTES[requested]


def _head_dim(text: dict[str, Any], heads: int) -> int:
    head_dim = text.get("head_dim")
    if head_dim is None:
        hidden_size = _positive_int(text.get("hidden_size"), "hidden_size")
        if hidden_size % heads:
            raise ValueError("SGLang automatic memory sizing requires hidden_size divisible by num_attention_heads")
        head_dim = hidden_size // heads
    return _positive_int(head_dim, "head_dim")


def _kv_bytes_per_token(config: dict[str, Any], kv_element_bytes: int, layout: _AttentionLayout | None = None) -> int:
    """Per-token KV bytes summed over every cache-bearing layer.

    Linear-attention layers hold no KV. Layers that read another layer's KV
    (Gemma 4 ``num_kv_shared_layers``) still get their own buffers. For a
    sliding-window hybrid this is the bytes per token when both pools hold the
    same number of tokens, which ``--swa-full-tokens-ratio 1.0`` makes exact.
    """
    text = config.get("text_config", config)
    if text.get("mamba_d_state") or text.get("index_topk"):
        raise ValueError("This SGLang cache layout requires an explicit mem_fraction_static")
    layout = layout or _attention_layout(text)
    if text.get("kv_lora_rank"):
        # Multi-head latent attention caches one compressed latent plus the
        # rotary key per layer (get_cell_size_per_token / calculate_mla_kv_cache_dim).
        latent = _positive_int(text.get("kv_lora_rank"), "kv_lora_rank")
        rope = _positive_int(text.get("qk_rope_head_dim"), "qk_rope_head_dim")
        return (latent + rope) * (layout.full + layout.sliding) * kv_element_bytes
    heads = _positive_int(text.get("num_attention_heads"), "num_attention_heads")
    kv_heads = _positive_int(text.get("num_key_value_heads", heads), "num_key_value_heads")
    head_dim = _head_dim(text, heads)
    v_head_dim = _positive_int(text.get("v_head_dim", head_dim), "v_head_dim")
    # Gemma 4 spells its sliding-window geometry in the base fields and its
    # full-attention geometry as ``global_*``; the engine swaps them when it
    # loads the config.
    full_heads = _positive_int(text.get("num_global_key_value_heads", kv_heads), "num_global_key_value_heads")
    full_dim = _positive_int(text.get("global_head_dim", head_dim), "global_head_dim")
    full_v_dim = full_dim if "global_head_dim" in text else v_head_dim
    full = full_heads * (full_dim + full_v_dim) * layout.full
    sliding = kv_heads * (head_dim + v_head_dim) * layout.sliding
    return (full + sliding) * kv_element_bytes


def _state_dtype_bytes(name: Any, source: str) -> int:
    if name not in _STATE_DTYPE_BYTES:
        raise ValueError(f"SGLang automatic memory sizing does not recognise {source}={name!r}")
    return _STATE_DTYPE_BYTES[name]


def _linear_state_bytes(config: dict[str, Any], linear_layers: int, args: Sequence[str], env: Mapping[str, str]) -> int:
    """Bytes of one recurrent-state slot of a Gated DeltaNet model.

    Mirrors ``Mamba2StateShape.create`` and ``mamba_cache_per_req``: per linear
    layer, a convolution window of ``(value_dim + 2 * key_dim, kernel - 1)`` in
    the conv dtype and a ``(value_heads, value_head_dim, key_head_dim)`` state
    in the SSM dtype.
    """
    text = config.get("text_config", config)
    key_heads = _positive_int(text.get("linear_num_key_heads"), "linear_num_key_heads")
    value_heads = _positive_int(text.get("linear_num_value_heads"), "linear_num_value_heads")
    key_dim = _positive_int(text.get("linear_key_head_dim"), "linear_key_head_dim")
    value_dim = _positive_int(text.get("linear_value_head_dim"), "linear_value_head_dim")
    kernel = _positive_int(text.get("linear_conv_kernel_dim"), "linear_conv_kernel_dim")
    conv_bytes = _state_dtype_bytes(env.get("SGLANG_MAMBA_CONV_DTYPE", "bfloat16"), "SGLANG_MAMBA_CONV_DTYPE")
    # --mamba-ssm-dtype sets the environment variable, which overrides the
    # checkpoint's own mamba_ssm_dtype; float32 is the engines' default.
    if (ssm := flag_value(args, "--mamba-ssm-dtype")) is not None:
        ssm_bytes = _state_dtype_bytes(ssm, "--mamba-ssm-dtype")
    elif (ssm := env.get("SGLANG_MAMBA_SSM_DTYPE")) is not None:
        ssm_bytes = _state_dtype_bytes(ssm, "SGLANG_MAMBA_SSM_DTYPE")
    else:
        ssm_bytes = _state_dtype_bytes(text.get("mamba_ssm_dtype", "float32"), "mamba_ssm_dtype")
    conv = (value_heads * value_dim + 2 * key_heads * key_dim) * (kernel - 1) * conv_bytes
    state = value_heads * value_dim * key_dim * ssm_bytes
    return (conv + state) * linear_layers


def _mamba_slots_per_request(args: Sequence[str], strategy_flag: str) -> int:
    if _has_flag(args, "--disable-radix-cache"):
        return 1
    overlap = not _has_flag(args, "--disable-overlap-schedule")
    strategy = flag_value(args, strategy_flag) or "auto"
    if strategy == "auto":
        strategy = _MAMBA_AUTO_STRATEGY.get(strategy_flag, "extra_buffer")
    if strategy == "extra_buffer":
        # Ping-pong buffers: two with the overlap scheduler, one without.
        return 5 if overlap else 4
    if strategy not in _MAMBA_STRATEGY_RATIO:
        raise ValueError(f"{strategy_flag} {strategy} requires an explicit mem_fraction_static")
    return _MAMBA_STRATEGY_RATIO[strategy]


@dataclass(frozen=True)
class LoraCapacity:
    """The LoRA pool a profile declares: slots per batch, rank and target modules."""

    max_per_batch: int
    max_rank: int | None
    target_modules: tuple[str, ...] | None


def _lora_pool_bytes(model_dir: Path, layers: int, capacity: LoraCapacity, element_bytes: int) -> int:
    """Bytes of the engine's LoRA buffers, allocated before KV profiling.

    For each normalized module ``m`` the engine holds an A buffer of
    ``(slots, stacked * rank, in)`` and a B buffer of ``(slots, out, rank)`` on
    every decoder layer (``LoRAMemoryPool``). Dimensions come from the
    checkpoint's own projection shapes, the widest layer standing in for all.
    """
    if capacity.max_rank is None or not capacity.target_modules or "all" in capacity.target_modules:
        raise ValueError("LoRA automatic memory sizing requires max_lora_rank and an explicit lora_target_modules list")
    modules = set()
    for name in capacity.target_modules:
        base = name.split(".")[-1]
        if base not in _LORA_MODULES:
            raise ValueError(f"LoRA target module {name!r} requires an explicit mem_fraction_static")
        modules.add(_LORA_MODULES[base])
    projections: dict[str, dict[str, list[int]]] = {}
    experts = False
    for name, tensor in _tensors(model_dir):
        if name.startswith("mtp.") or any(marker in name for marker in _TOWER_MARKERS):
            continue
        if ".experts." in name:
            experts = True
        elif match := _PROJECTION.search(name):
            projections.setdefault(match["layer"], {})[match["module"]] = tensor["shape"]
    if experts and modules & {"gate_up_proj", "down_proj"}:
        raise ValueError("LoRA on mixture-of-experts layers requires an explicit mem_fraction_static")
    widths: dict[str, int] = {}

    def widen(module: str, stacked: int, shapes: list[list[int]]) -> None:
        width = stacked * shapes[0][1] + sum(shape[0] for shape in shapes)
        widths[module] = max(widths.get(module, 0), width)

    for layer in projections.values():
        if "qkv_proj" in layer:
            widen("qkv_proj", 3, [layer["qkv_proj"]])
        elif "q_proj" in layer:
            # A checkpoint that reuses K as V (Gemma 4 global layers) omits v_proj.
            kv = layer.get("k_proj") or layer.get("v_proj") or [0, 0]
            widen("qkv_proj", 3, [layer["q_proj"], kv, layer.get("v_proj", kv)])
        if "o_proj" in layer:
            widen("o_proj", 1, [layer["o_proj"]])
        if "gate_up_proj" in layer:
            widen("gate_up_proj", 2, [layer["gate_up_proj"]])
        elif "gate_proj" in layer and "up_proj" in layer:
            widen("gate_up_proj", 2, [layer["gate_proj"], layer["up_proj"]])
        if "down_proj" in layer:
            widen("down_proj", 1, [layer["down_proj"]])
    missing = modules - widths.keys()
    if missing:
        raise ValueError(f"LoRA target modules {sorted(missing)} have no matching checkpoint projection")
    return element_bytes * capacity.max_per_batch * capacity.max_rank * layers * sum(widths[m] for m in modules)


def device_memory(device_index: int) -> tuple[int, int]:
    """Return current free and total bytes, including other CUDA processes."""
    import torch

    return torch.cuda.mem_get_info(device_index)


def _request_table_bytes(args: Sequence[str], token_capacity: int, context_length: int) -> int:
    """Bytes of the engine's request-to-token table, allocated outside the static fraction.

    One int32 row of ``context_length + 4`` per running request plus one;
    without ``--max-running-requests`` the engines admit
    ``min(clamp(tokens / context * 512, 2048, 4096), tokens // 2)``.
    """
    running = _flag_int(args, "--max-running-requests")
    if running is None:
        estimated = min(max(int(token_capacity / context_length * 512), 2048), 4096)
        running = min(estimated, token_capacity // 2)
    return (running + 1) * (context_length + 4) * 4


@dataclass(frozen=True)
class SGLangReservation:
    """What one engine needs, derived from checkpoint metadata.

    ``static_bytes`` is held inside the engine's static fraction for its
    lifetime and ``runtime_bytes`` outside it, beside CUDA graphs.
    ``load_bytes`` is the free memory weight loading needs when it peaks above
    that, as with online quantization; zero otherwise. ``launch_args`` pin the
    pools the reservation assumes.
    """

    static_bytes: int
    load_bytes: int = 0
    launch_args: tuple[str, ...] = ()
    runtime_bytes: int = RUNTIME_HEADROOM_BYTES


@dataclass
class SGLangMemoryBudget:
    model_name_or_path: str
    revision: str | None
    compute_precision: str
    max_seq_length: int
    kv_budget_tokens: int | None = None
    mem_fraction_static: float | None = None
    tensor_parallel_size: int = 1
    speculative: bool = False
    lora: LoraCapacity | None = None
    # The profile's raw engine flags. Read, never rewritten: they decide the
    # KV dtype, online quantization and the recurrent-state pool layout.
    launch_options: Sequence[str] = ()
    # Environment the profile adds to the engine process.
    launch_env: Mapping[str, str] = field(default_factory=dict)
    # The engine's spelling of the mamba radix-cache strategy flag; it also
    # selects that engine's ``auto`` resolution.
    mamba_strategy_flag: str = "--mamba-scheduler-strategy"
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
    def reservation(self) -> SGLangReservation:
        # Tensor parallelism adds collective buffers that depend on topology,
        # not metadata. Speculative drafts load their weights, KV and head
        # copies inside the static fraction on one pinned engine and after it
        # on the other, so neither placement can be assumed.
        if self.tensor_parallel_size != 1 or self.speculative:
            raise ValueError("Tensor parallelism and speculative decoding require an explicit mem_fraction_static")
        args = tuple(self.launch_options)
        model_dir = _model_directory(self.model_name_or_path, self.revision)
        config = json.loads((model_dir / "config.json").read_text())
        text = config.get("text_config", config)
        element_bytes = _PRECISION_BYTES[self.compute_precision]

        quantization = _quantization_config(config).get("quant_method")
        online = flag_value(args, "--quantization")
        if quantization is not None and quantization not in _SERIALIZED_QUANTIZATION:
            raise ValueError(f"{quantization} checkpoints require an explicit mem_fraction_static")
        if quantization is None and online is not None and online not in _ONLINE_QUANTIZATION:
            raise ValueError(f"--quantization {online} requires an explicit mem_fraction_static")
        weights, peak = _weight_bytes(
            model_dir,
            element_bytes,
            serialized_quantization=quantization is not None,
            online_quantization=quantization is None and online is not None,
        )

        layout = _attention_layout(text)
        kv = _kv_bytes_per_token(config, _kv_element_bytes(config, args, element_bytes), layout) * (
            self.token_capacity + TOKEN_ALIGNMENT
        )
        pools, launch = self._state_pool(config, layout, args)
        if self.lora is not None:
            pools += _lora_pool_bytes(
                model_dir, _positive_int(text.get("num_hidden_layers"), "num_hidden_layers"), self.lora, element_bytes
            )
        if layout.sliding and layout.full and flag_value(args, "--swa-full-tokens-ratio") is None:
            # 0.5.20 gives the sliding-window pool 0.8 of the full pool's tokens
            # by default and 0.5.10 spreads the same bytes differently. At 1.0
            # both allocate the full token budget in each pool, which is what
            # the reservation above holds.
            launch = (*launch, "--swa-full-tokens-ratio", "1.0")
        # The child initializes its CUDA context before measuring free memory.
        # Keep this allowance inside its static fraction; cap the actual KV pool.
        static = math.ceil(weights * WEIGHT_ALLOWANCE) + kv + pools + _server.LOAD_HEADROOM_BYTES
        load = math.ceil(peak * WEIGHT_ALLOWANCE) + _server.LOAD_HEADROOM_BYTES if peak > weights else 0
        runtime = RUNTIME_HEADROOM_BYTES + _request_table_bytes(
            args, self.token_capacity, _positive_int(self.max_seq_length, "max_seq_length")
        )
        return SGLangReservation(static, load, launch, runtime)

    def _state_pool(
        self, config: dict[str, Any], layout: _AttentionLayout, args: Sequence[str]
    ) -> tuple[int, tuple[str, ...]]:
        """Size and pin the recurrent-state pool of a hybrid linear-attention model.

        Without a pinned size the engines give that pool a share of the
        post-load memory, starving the KV pool. SIE pins it to the state slots
        ``--max-running-requests`` needs under the radix strategy in effect,
        and reserves one padding slot more, as the engine allocates.
        """
        if not layout.linear:
            return 0, ()
        slot = _linear_state_bytes(config, layout.linear, args, self.launch_env)
        slots = _flag_int(args, "--max-mamba-cache-size")
        if slots is not None:
            return (slots + 1) * slot, ()
        running = _flag_int(args, "--max-running-requests")
        if running is None:
            raise ValueError(
                "Hybrid linear-attention models size their recurrent-state pool from --max-running-requests; "
                "add it to extra_launch_args or set mem_fraction_static"
            )
        slots = running * _mamba_slots_per_request(args, self.mamba_strategy_flag)
        return (slots + 1) * slot, ("--max-mamba-cache-size", str(slots))

    @property
    def static_bytes(self) -> int:
        return self.reservation.static_bytes

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
        reservation = self.reservation
        resident = reservation.static_bytes + reservation.runtime_bytes
        self.reserved_bytes = resident
        return max(resident, reservation.load_bytes)

    def launch_args(self, device_index: int) -> list[str]:
        if self.mem_fraction_static is not None:
            return ["--mem-fraction-static", str(self.mem_fraction_static)]
        free, total = device_memory(device_index)
        required = self.required_bytes(device_type="cuda", device_total_bytes=total)
        if required is None or free < required:
            raise RuntimeError(f"CUDA out of memory: SGLang needs {required} bytes free, but only {free} are available")
        reservation = self.reservation
        # Both pinned engines profile KV from post-load free minus pre-load
        # free * (1 - fraction), not a fraction of total VRAM.
        fraction = reservation.static_bytes / free
        logger.info(
            "SGLang memory reservation for %s: %.2f GiB resident, %.2f GiB to load, KV capacity %d tokens, "
            "mem_fraction_static=%.6f",
            self.model_name_or_path,
            self.reserved_bytes / GIB,
            required / GIB,
            self.token_capacity,
            fraction,
        )
        return [
            "--mem-fraction-static",
            str(fraction),
            "--max-total-tokens",
            str(self.token_capacity),
            *reservation.launch_args,
        ]
