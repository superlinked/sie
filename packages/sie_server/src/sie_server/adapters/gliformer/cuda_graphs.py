"""Opt-in CUDA graphs for GLiFormer's text encoder, with eager task heads.

Only the ``TextTransformer`` token-id/mask forward is captured. Word gathering,
the packed word RNN, task heads, output checks and decoding stay eager. Layout,
image, packed and otherwise nonstandard encoder inputs also stay eager.

``exact`` records a repeated input shape without padding. Graphs hold at most
2,048 tokens (1,024 for encoders wider than 768), and at most 64 shapes. These
are admission bounds, not a performance or numerical-equivalence claim.
Bucketed execution is not supported: padding failed pretrained output parity.

Recording follows the process-wide lock, 10% free-memory headroom and 4%
device-memory budget. A token bucket permits 16 recordings initially, then
one per two seconds. Capture failures fall back eagerly; OOM drops graphs
and pauses recording for a minute. The cache can be released independently
of model weights. Captures that exceed the retained-memory budget are dropped.
Each shape owns a separate graph pool so shapes can replay in arbitrary order.
After capture, allocator snapshots count the pool's complete segments; the
larger of that total and the driver-memory delta is charged along with static
buffers. Missing or inconsistent accounting rejects the optional capture.

Each replay runs on a private stream and copies its result before returning.
The caller owns that copy: concurrent callers cannot overwrite hidden states
while eager heads still read them. Clear/unload waits for queued replay work
before releasing graph storage. The relative-position override is local to
the instance and recording thread; close restores the original attributes.
Layout attention's CPU-computed scale is copied to the device before capture.
Only the recording thread uses that constant through a temporary instance
override; the upstream method and module globals remain unchanged otherwise.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any

import torch

from sie_server.adapters._cuda_graphs import RECORDING_LOCK, CudaGraphStats, free_memory, has_headroom, memory_budget
from sie_server.core.oom import is_oom_error

logger = logging.getLogger(__name__)

GRAPH_MODES = ("off", "exact")
_MAX_GRAPHS = 64
_MAX_SIGHTINGS = 4096
_RECORDING_BURST = 16
_SECONDS_PER_RECORDING = 2.0
_OOM_PAUSE_SECONDS = 60.0
_MAX_FAILURES = 3
_MISSING = object()
Key = tuple[int, int]


def _out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or is_oom_error(error)


def _capture_device_bytes(graph: Any, device: torch.device, free_before: int) -> int:
    """Count the private pool outside capture, without relying on a device delta."""
    pool = graph.pool()
    # Torch 2.9's pool-argument binding decrements borrowed tuple references.
    # Snapshot all pools and filter in Python to avoid that native-code defect.
    snapshot = torch.cuda.memory_snapshot()
    if not isinstance(snapshot, list) or any(not isinstance(segment, dict) for segment in snapshot):
        raise RuntimeError("GLiFormer graph memory snapshot is unavailable")
    segments = [segment for segment in snapshot if segment.get("segment_pool_id") == pool]
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not segments:
        raise RuntimeError("GLiFormer graph memory snapshot is unavailable")
    addresses: set[int] = set()
    pool_bytes = 0
    for segment in segments:
        size, address = segment.get("total_size"), segment.get("address")
        if (
            segment.get("segment_pool_id") != pool
            or segment.get("device") != index
            or type(size) is not int
            or size <= 0
            or type(address) is not int
            or address <= 0
            or address in addresses
        ):
            raise RuntimeError("GLiFormer graph memory snapshot is inconsistent")
        addresses.add(address)
        pool_bytes += size
    return max(pool_bytes, 0, free_before - free_memory(device))


def graph_shape(rows: int, length: int, *, mode: str, window: int, max_tokens: int) -> Key | None:
    """Admit a bounded exact shape without adding rows or token positions."""
    if mode != "exact" or rows < 1 or not 0 < length <= window:
        return None
    return (rows, length) if rows * length <= max_tokens else None


def unsupported_reason(transformer: Any, device: str | torch.device) -> str | None:
    """Admit only the actual text encoder implementations verified for capture."""
    if not str(device).startswith("cuda"):
        return "CUDA graphs require a CUDA device"

    # GLiFormer is optional and is imported by the adapter with its AutoModel
    # registration guard before this function is used.
    from gliformer.backbones.deberta_2d import LayoutDebertaModel, flash_kernels_available
    from gliformer.encoders.text import TextTransformer
    from transformers import DebertaV2Model

    if type(transformer) is not TextTransformer:
        return "the text transformer implementation is unsupported"
    encoder = transformer.model
    if type(encoder) not in (DebertaV2Model, LayoutDebertaModel):
        return "the backbone implementation is unsupported"
    if type(encoder) is LayoutDebertaModel and encoder.encoder.attn_kernel != "eager" and flash_kernels_available():
        return "the layout backbone uses flash attention"
    if transformer.training or encoder.training:
        return "the text encoder is in training mode"
    if getattr(encoder, "z_steps", 0) > 1:
        return "repeated query-state encoding is unsupported"
    if not callable(getattr(encoder.encoder, "get_rel_pos", None)):
        return "the backbone has no relative-position hook"
    return None


@dataclass
class _Graph:
    graph: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    hidden: torch.Tensor
    device_bytes: int
    attention_scales: tuple[torch.Tensor, ...] = ()


class EncoderGraphRunner:
    """Own one model's encoder forward override and bounded graph cache."""

    def __init__(
        self,
        transformer: Any,
        *,
        mode: str,
        window: int,
        pad_token_id: int,
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode != "exact" or window < 1:
            raise ValueError("GLiFormer graph runner needs exact mode and a positive window")
        self._transformer = transformer
        self._encoder = transformer.model
        self._layers = self._encoder.encoder
        self._forward = transformer.forward
        self._build_relative_pos = self._layers.get_rel_pos
        self._forward_attribute = transformer.__dict__.get("forward", _MISSING)
        self._relative_attribute = self._layers.__dict__.get("get_rel_pos", _MISSING)
        self._mode, self._window, self._pad_token_id = mode, window, pad_token_id
        self._name = name
        self._embedding = self._encoder.get_input_embeddings().weight
        self._device, self._dtype = self._embedding.device, self._embedding.dtype
        self._embedding_pointer = self._embedding.data_ptr()
        self._hidden_size = int(self._encoder.config.hidden_size)
        self._max_tokens = 1024 if self._hidden_size > 768 else 2048
        self._graphs: OrderedDict[Key, _Graph] = OrderedDict()
        self._seen: OrderedDict[Key, None] = OrderedDict()
        self._relative_pos: dict[int, torch.Tensor | None] = {}
        self._failed: set[Key] = set()
        self._full = False
        self._closed = False
        self._disabled = False
        self._clock = clock
        self._credit = float(_RECORDING_BURST)
        self._credited_at = clock()
        self._pause_until = 0.0
        self._stream: torch.cuda.Stream | None = None
        self._local = threading.local()
        self._lock = threading.RLock()
        self.stats = CudaGraphStats()
        self._wrapper = self._call

        def relative_position(hidden_states: torch.Tensor, query_states: Any = None, relative_pos: Any = None) -> Any:
            if relative_pos is None and getattr(self._local, "recording", False):
                return self._local.relative_pos
            return self._build_relative_pos(hidden_states, query_states, relative_pos)

        self._relative_wrapper = relative_position
        self._layers.get_rel_pos = relative_position
        transformer.forward = self._wrapper

    def key(self, rows: int, length: int) -> Key | None:
        return graph_shape(rows, length, mode=self._mode, window=self._window, max_tokens=self._max_tokens)

    def _inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
        # Do not silently discard a layout, packing, embedding, attention or
        # output-mode argument, even when its value looks like a default.
        if len(args) > 1 or set(kwargs) - {"input_ids", "attention_mask"} or (args and "input_ids" in kwargs):
            return None
        ids = args[0] if args else kwargs.get("input_ids")
        mask = kwargs.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
            return None
        if ids.ndim != 2 or mask.shape != ids.shape or ids.dtype != torch.long or mask.dtype != torch.long:
            return None
        if ids.device != self._device or mask.device != self._device or ids.device.type != "cuda":
            return None
        if not torch.is_inference_mode_enabled() or torch.is_autocast_enabled("cuda"):
            return None
        if self._transformer.model is not self._encoder or self._transformer.training or self._encoder.training:
            return None
        weight = self._encoder.get_input_embeddings().weight
        if (
            weight is not self._embedding
            or weight.device != self._device
            or weight.dtype != self._dtype
            or weight.data_ptr() != self._embedding_pointer
        ):
            return None
        return ids, mask

    def _call(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        inputs = self._inputs(args, kwargs)
        if inputs is None:
            self._eager("unsupported_input")
        else:
            key = self.key(*inputs[0].shape)
            if key is None:
                self._eager("shape_limit")
            else:
                result = self.run(*inputs, key)
                if result is not None:
                    return result
        return self._forward(*args, **kwargs)

    def _eager(self, reason: str) -> None:
        self.stats.eager[reason] = self.stats.eager.get(reason, 0) + 1

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, key: Key) -> torch.Tensor | None:
        """Return a caller-owned hidden-state tensor, or None for eager execution."""
        with self._lock:
            if self._closed or self._disabled:
                self._eager("disabled")
                return None
            entry = self._graphs.get(key)
            if entry is not None:
                self._graphs.move_to_end(key)
                try:
                    result = self._replay(entry, input_ids, attention_mask)
                except Exception as exc:
                    if _out_of_memory(exc):
                        self._recording_failed(key, exc)
                    raise
                self.stats.replayed += 1
                return result
            reason = self._why_not_record(key, input_ids.device)
            if reason is not None:
                self._eager(reason)
                return None
            if not RECORDING_LOCK.acquire(blocking=False):
                self._eager("busy")
                return None
            try:
                self._credit -= 1
                try:
                    entry = self._capture(key, input_ids.device)
                    self._graphs[key] = entry
                    if self.held_bytes > memory_budget(input_ids.device):
                        del entry
                        self.clear()
                        self._full = True
                        self.stats.drops += 1
                        self._eager("budget_full")
                        return None
                    result = self._replay(entry, input_ids, attention_mask)
                except Exception as exc:  # noqa: BLE001 -- capture is optional; the real forward remains eager
                    self._recording_failed(key, exc)
                    self._eager("recording_failed")
                    return None
                self.stats.recorded += 1
                if len(self._graphs) > _MAX_GRAPHS and self._stream is not None:
                    # Old graphs may still have queued kernels on the private
                    # stream. Only eviction, not ordinary replay, waits here.
                    self._stream.synchronize()
                while len(self._graphs) > _MAX_GRAPHS:
                    self._graphs.popitem(last=False)
                self._prune_tables()
                return result
            finally:
                RECORDING_LOCK.release()

    def _why_not_record(self, key: Key, device: torch.device) -> str | None:
        now = self._clock()
        self._credit = min(_RECORDING_BURST, self._credit + max(0.0, now - self._credited_at) / _SECONDS_PER_RECORDING)
        self._credited_at = now
        if key in self._failed:
            return "failed_shape"
        if self._full:
            return "budget_full"
        if now < self._pause_until or self._credit < 1:
            return "recording_paused"
        if self._mode == "exact" and key not in self._seen:
            self._seen[key] = None
            if len(self._seen) > _MAX_SIGHTINGS:
                self._seen.popitem(last=False)
            return "first_shape"
        if key in self._seen:
            self._seen.move_to_end(key)
        if not has_headroom(device):
            return "no_headroom"
        return None

    def _recording_failed(self, key: Key, error: BaseException) -> None:
        self.clear()
        self.stats.drops += 1
        if _out_of_memory(error):
            self._pause_until = self._clock() + _OOM_PAUSE_SECONDS
        else:
            self._failed.add(key)
            self.stats.recording_failures += 1
            self._disabled = len(self._failed) >= _MAX_FAILURES
        logger.warning(
            "GLiFormer encoder graph for %s, shape %s, failed; using eager execution", self._name, key, exc_info=error
        )

    def _prune_tables(self) -> None:
        lengths = {key[1] for key in self._graphs}
        self._relative_pos = {length: value for length, value in self._relative_pos.items() if length in lengths}

    @contextlib.contextmanager
    def _recording(self, relative_pos: torch.Tensor | None) -> Iterator[None]:
        self._local.recording, self._local.relative_pos = True, relative_pos
        try:
            yield
        finally:
            self._local.recording, self._local.relative_pos = False, None

    @contextlib.contextmanager
    def _layout_attention_scales(self, device: torch.device) -> Iterator[tuple[torch.Tensor, ...]]:
        # GLiFormer is optional and already imported for the admitted encoder.
        from gliformer.backbones.deberta_2d import LayoutDisentangledSelfAttention

        original = LayoutDisentangledSelfAttention.forward
        modules = [module for module in self._encoder.modules() if type(module) is LayoutDisentangledSelfAttention]
        if not modules:
            yield ()
            return
        scales: dict[tuple[int, int], torch.Tensor] = {}
        helper = original.__globals__["scaled_size_sqrt"]
        for module in modules:
            bound = module.forward
            if not isinstance(bound, MethodType) or bound.__self__ is not module or bound.__func__ is not original:
                raise RuntimeError("GLiFormer layout attention has an unsupported forward override")
            factor = 1 + ("c2p" in module.pos_att_type) + ("p2c" in module.pos_att_type)
            key = (module.attention_head_size, factor)
            if key not in scales:
                # Preserve upstream CPU float32 sqrt, device transfer, then the
                # forward's dtype conversion. A Python float sqrt can round
                # differently. This transfer must finish before capture.
                shape = torch.empty((0, key[0]), dtype=torch.float32, device="cpu")
                scales[key] = helper(shape, factor).to(device=device)

        def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
            if query_layer.device != device:
                raise RuntimeError("GLiFormer layout attention changed devices during capture")
            return scales[(query_layer.size(-1), scale_factor)]

        # Reuse the upstream code rather than fork its attention implementation.
        # A private globals dictionary changes this one helper only; other
        # methods (including relative attention) still use their original code.
        cloned = FunctionType(
            original.__code__,
            {**original.__globals__, "scaled_size_sqrt": scaled_size_sqrt},
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        cloned.__kwdefaults__ = original.__kwdefaults__
        local = self._local

        def forward(module: Any, *args: Any, **kwargs: Any) -> Any:
            selected = cloned if getattr(local, "recording", False) else original
            return selected(module, *args, **kwargs)

        attributes: list[tuple[Any, Any]] = []
        try:
            for module in modules:
                attributes.append((module, module.__dict__.get("forward", _MISSING)))
                module.forward = MethodType(forward, module)
            yield tuple(scales.values())
        finally:
            for module, attribute in reversed(attributes):
                if attribute is _MISSING:
                    if "forward" in module.__dict__:
                        delattr(module, "forward")
                else:
                    module.forward = attribute

    def _capture(self, key: Key, device: torch.device) -> _Graph:
        rows, length = key
        current = torch.cuda.current_stream(device)
        with torch.inference_mode():
            ids = torch.full(key, self._pad_token_id, dtype=torch.long, device=device)
            mask = torch.ones(key, dtype=torch.long, device=device)
            hidden = torch.empty((rows, length, self._hidden_size), dtype=self._dtype, device=device)
            if length not in self._relative_pos:
                self._relative_pos[length] = self._build_relative_pos(torch.empty((1, length, 1), device=device))
            relative_pos = self._relative_pos[length]
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        stream = self._stream
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode(), self._layout_attention_scales(device) as attention_scales:
                stream.wait_stream(current)
                with torch.cuda.stream(stream), self._recording(relative_pos):
                    if getattr(self._local, "warmed_stream", None) is not stream:
                        # Each serving thread needs its own cuBLAS state initialized
                        # outside capture, even when another thread warmed this model.
                        self._forward(input_ids=ids[:1], attention_mask=mask[:1])
                        self._local.warmed_stream = stream
                    free_before = free_memory(device)
                    # Sharing pools requires capture-order replay. Requests can
                    # revisit shapes in any order, so every shape owns its pool.
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        output = self._forward(input_ids=ids, attention_mask=mask)
                        if (
                            not isinstance(output, torch.Tensor)
                            or output.shape != hidden.shape
                            or output.dtype != hidden.dtype
                        ):
                            raise RuntimeError("GLiFormer encoder graph returned an unexpected hidden-state tensor")
                        hidden.copy_(output)
                        del output
                    except BaseException:
                        with contextlib.suppress(Exception):
                            graph.capture_end()
                        raise
                    graph.capture_end()
                    device_bytes = _capture_device_bytes(graph, device, free_before)
        finally:
            current.wait_stream(stream)
        return _Graph(graph, ids, mask, hidden, device_bytes, attention_scales)

    def _replay(self, entry: _Graph, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        stream = self._stream
        if stream is None:
            raise RuntimeError("GLiFormer graph has no replay stream")
        current = torch.cuda.current_stream(input_ids.device)
        stream.wait_stream(current)
        rows, length = input_ids.shape
        with torch.inference_mode(), torch.cuda.stream(stream):
            entry.input_ids[:, :length].copy_(input_ids)
            entry.attention_mask[:, :length].copy_(attention_mask)
            if length < entry.input_ids.shape[1]:
                entry.input_ids[:, length:].fill_(self._pad_token_id)
                entry.attention_mask[:, length:].zero_()
            entry.graph.replay()
            output = entry.hidden[:rows, :length].clone()
        current.wait_stream(stream)
        output.record_stream(current)
        return output

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def held_bytes(self) -> int:
        tensors = sum(
            entry.device_bytes
            + entry.input_ids.nbytes
            + entry.attention_mask.nbytes
            + entry.hidden.nbytes
            + sum(scale.nbytes for scale in entry.attention_scales)
            for entry in self._graphs.values()
        )
        tables = sum(table.nbytes for table in self._relative_pos.values() if table is not None)
        return tensors + tables

    def clear(self) -> None:
        """Wait for scheduled graph work, then drop optional device storage."""
        with self._lock:
            if self._stream is not None:
                self._stream.synchronize()
            self._graphs.clear()
            self._relative_pos.clear()
            self._full = False

    def close(self) -> None:
        """Release the cache and restore the original instance attribute layout."""
        with self._lock:
            if self._closed:
                return
            self.clear()
            for target, name, original in (
                (self._transformer, "forward", self._forward_attribute),
                (self._layers, "get_rel_pos", self._relative_attribute),
            ):
                if original is _MISSING:
                    delattr(target, name)
                else:
                    setattr(target, name, original)
            self._closed = True
            self._stream = None
