"""CUDA graphs for the GLiClass forward pass.

A GLiClass forward on a DeBERTa encoder launches about a thousand small
kernels, so at small batch sizes the GPU waits on the CPU that launches them.
A CUDA graph records the launches of one input shape once and replays them
with a single call.

Graphs are an operator setting, fixed when the model loads
(``adapter_options.loadtime.cuda_graphs``); a request can only opt out. A
graph is recorded per (batch size, sequence length, class slots) and kept in
a bounded least-recently-used cache. Forwards over 2,048 tokens run eagerly:
the GPU, not kernel launches, bounds them. Two modes choose the sequence
length:

- ``exact`` records the request's own length. Replay launches the kernels
  eager execution launches, so scores are bit-identical to eager. A shape is
  recorded the second time it is seen, so lengths seen once cost nothing.
- ``bucketed`` pads the length up to a bucket (a multiple of 32 tokens at a
  512-token window, 64 at 1,024), so a few graphs cover every length. Padding
  is masked, but a longer sequence rounds fp16 sums differently, as when
  requests are batched together: up to 0.006 in probability on
  gliclass-large-v1.0 and 0.023 on gliclass-multilang-mini in our tests.

While a graph is being recorded, PyTorch's caching allocator will not free
cached blocks to satisfy another allocation, so another model on the same GPU
can hit an out-of-memory error it would otherwise have avoided. Recording is
therefore kept rare and short:

- at most one recording at a time in the process, across all models; a
  forward that finds another recording in progress runs eagerly;
- no recording while less than a tenth of the device's memory is free;
- a runner records at most 16 graphs at once, then one per 2 seconds.

A runner's graphs hold device memory the server does not attribute to the
model: their shared pool, which also keeps what evicted graphs used, the
driver's copy of each recorded graph (about 8 MB for a DeBERTa-large
forward), and the relative-position tables and static buffers they read. The
runner adds up the device memory each recording takes, plus those tensors.
Past 4% of the device's memory it drops every graph, returns the memory to the
device and starts again.

Recording needs a forward that never waits on the host. Two parts of the
gliclass DeBERTa forward do: the relative-position table copies a CPU scalar
to the GPU, and segment ids (instruct and Opir multitask models) read each
row's positions with ``.item()``. While recording, the runner hands the
encoder a table precomputed for the length, shared by the graphs of that
length, and computes segment ids with tensor operations; both give the same
integers.

Replays never overlap (a lock serializes them) and every result is copied out
before the next replay.
Anything unsupported, larger than the token bound, or failing to record runs
eagerly; a recording failure turns the runner off.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from sie_server.core.oom import is_oom_error

logger = logging.getLogger(__name__)

GraphMode = Literal["off", "exact", "bucketed"]
GRAPH_MODES: tuple[GraphMode, ...] = ("off", "exact", "bucketed")
# Recorded graphs per runner. Their activations share one pool; each also
# holds its static inputs and output (up to 48 KB at 2,048 tokens).
_MAX_GRAPHS = 64
# Shapes whose sightings ``exact`` mode counts.
_MAX_SIGHTINGS = 4096
# ``exact`` mode records a shape the time it is seen this many times.
_EXACT_RECORD_AT = 2
# Tokens (batch x padded length) a graph may hold. Past about that many tokens
# the GPU, not kernel launches, bounds a forward, so a graph saves little and
# would pin a larger memory pool.
_MAX_GRAPH_TOKENS = 2048
# A runner may record this many graphs at once, then one per this many
# seconds, however its forwards repeat: recording costs a forward, and while
# it runs the allocator cannot free cached memory for other models.
_RECORDING_BURST = 16
_SECONDS_PER_RECORDING = 2.0
# Recording waits until at least this share of the device's memory is free.
_RECORDING_HEADROOM = 0.1
# After a recording runs out of memory, the runner records nothing for this long.
_OOM_COOL_DOWN_SECONDS = 60.0
# One recording at a time in the process, across every model's runner.
_RECORDING_LOCK = threading.Lock()
# Device memory a runner's graphs may hold (their recordings, relative-position
# tables and static buffers), as a share of the device's memory, before it
# drops them and starts again.
_MEMORY_BUDGET_SHARE = 0.04
# Encoders whose forward records cleanly once the two host reads are replaced.
_ENCODERS = frozenset({"deberta-v2"})
# Poolings whose pooled vector ignores padded positions, so ``bucketed`` mode
# pads only the length.
_PADDING_SAFE_POOLINGS = frozenset({"first", "pass"})

Key = tuple[int, int, int]


def unsupported_reason(model: Any, device: str | torch.device) -> str | None:
    """Why ``model`` cannot use CUDA graphs on ``device``; None when it can."""
    if not str(device).startswith("cuda"):
        return "CUDA graphs need a CUDA device"
    config = getattr(model, "config", None)
    if getattr(config, "architecture_type", None) != "uni-encoder":
        return "only uni-encoder GLiClass models are supported"
    encoder_type = getattr(getattr(config, "encoder_config", None), "model_type", None)
    if encoder_type not in _ENCODERS:
        return f"the {encoder_type} encoder is not supported"
    encoder = getattr(getattr(getattr(model, "model", None), "encoder_model", None), "encoder", None)
    if encoder is None or not hasattr(encoder, "get_rel_pos"):
        return "the encoder has no relative-position hook"
    return None


def bucket_width(max_length: int) -> int:
    """Sequence-length bucket for ``bucketed`` mode: 32 tokens at a 512 window, 64 at 1,024."""
    return max(16, max_length // 16)


class _RecordingError(Exception):
    """Recording a graph failed after the eager forward had answered the call."""

    def __init__(self, cause: BaseException, logits: torch.Tensor) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.logits = logits


@dataclass
class _Graph:
    graph: Any
    inputs: dict[str, torch.Tensor]
    output: torch.Tensor
    # Device memory the recording took: pool growth and the recorded graph.
    device_bytes: int = 0


class CudaGraphRunner:
    """Records and replays the forward pass of one GLiClass model, per input shape."""

    def __init__(
        self,
        model: Any,
        *,
        pad_token_id: int,
        max_length: int,
        pad_token_type_id: int = 0,
        max_graphs: int = _MAX_GRAPHS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "token_type_ids": pad_token_type_id}
        self._max_length = max_length
        self._bucket = bucket_width(max_length)
        self._max_tokens = _MAX_GRAPH_TOKENS
        self._max_graphs = max_graphs
        self._graphs: OrderedDict[Key, _Graph] = OrderedDict()
        self._sightings: OrderedDict[Key, int] = OrderedDict()
        self._lock = threading.RLock()
        self._stream: torch.cuda.Stream | None = None
        self._disabled = False
        self._clock = clock
        # Device memory recordings have taken since the graphs were last
        # dropped; evicting a graph does not give its share of the pool back.
        self._device_bytes = 0
        self._recording_credit = float(_RECORDING_BURST)
        self._credited_at = clock()
        # Relative-position tables of the recorded lengths; a graph reads its
        # length's table on every replay.
        self._relative_pos: dict[int, torch.Tensor] = {}
        # Set only while recording: the relative-position table the encoder uses.
        self._recording_relative_pos: torch.Tensor | None = None
        self._recording = False
        config = model.config
        self._padding_safe = config.pooling_strategy in _PADDING_SAFE_POOLINGS and not getattr(
            config, "use_lstm", False
        )
        self._install_hooks()

    # -- hooks -------------------------------------------------------------

    def _install_hooks(self) -> None:
        inner = self._model.model
        encoder = inner.encoder_model.encoder
        get_rel_pos = encoder.get_rel_pos
        self._build_relative_pos = get_rel_pos

        def recording_get_rel_pos(
            hidden_states: torch.Tensor, query_states: Any = None, relative_pos: Any = None
        ) -> Any:
            if self._recording and relative_pos is None:
                return self._recording_relative_pos
            return get_rel_pos(hidden_states, query_states, relative_pos)

        encoder.get_rel_pos = recording_get_rel_pos
        if getattr(self._model.config, "use_segment_embeddings", False):
            create_segment_ids = inner._create_segment_ids
            config = self._model.config

            def recording_segment_ids(input_ids: torch.Tensor) -> torch.Tensor:
                if not self._recording:
                    return create_segment_ids(input_ids)
                return segment_ids(input_ids, config)

            inner._create_segment_ids = recording_segment_ids

    # -- policy ------------------------------------------------------------

    def key(self, batch: int, length: int, classes: int, mode: GraphMode) -> Key | None:
        """The graph a forward of this shape would replay; None when it runs eagerly."""
        if mode == "bucketed" and self._padding_safe:
            length = min(self._max_length, math.ceil(length / self._bucket) * self._bucket)
        if batch * length > self._max_tokens or length > self._max_length:
            return None
        return (batch, length, classes)

    def _should_record(self, key: Key, mode: GraphMode) -> bool:
        if mode != "exact":
            return True
        seen = self._sightings.pop(key, 0) + 1
        self._sightings[key] = seen
        while len(self._sightings) > _MAX_SIGHTINGS:
            self._sightings.popitem(last=False)
        return seen >= _EXACT_RECORD_AT

    # -- entry point -------------------------------------------------------

    def run(self, inputs: dict[str, torch.Tensor], max_num_classes: int, mode: GraphMode) -> torch.Tensor | None:
        """Logits for ``inputs`` from a graph, or None to run the forward eagerly."""
        if mode == "off" or self._disabled:
            return None
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.dim() != 2 or set(inputs) - set(self._pad_values):
            return None
        batch, length = input_ids.shape
        key = self.key(batch, length, max_num_classes, mode)
        if key is None:
            return None
        with self._lock:
            now = self._clock()
            elapsed, self._credited_at = now - self._credited_at, now
            self._recording_credit = min(
                float(_RECORDING_BURST), self._recording_credit + elapsed / _SECONDS_PER_RECORDING
            )
            entry = self._graphs.get(key)
            if entry is not None:
                self._graphs.move_to_end(key)
                return self._replay(entry, inputs, length)
            if not self._should_record(key, mode) or self._recording_credit < 1:
                return None
            if not self._has_headroom(input_ids.device):
                return None
            if not _RECORDING_LOCK.acquire(blocking=False):
                return None  # another model is recording
            try:
                self._recording_credit -= 1
                try:
                    entry, logits = self._record(key, inputs)
                except _RecordingError as failure:
                    # The eager forward has answered this call; only the graph
                    # failed, so the answer stands and no OOM recovery is needed.
                    self.clear()
                    if is_oom_error(failure.cause):
                        logger.info(
                            "GLiClass CUDA graph recording ran out of memory; recording paused for %d s",
                            _OOM_COOL_DOWN_SECONDS,
                        )
                        self._recording_credit = 1 - _OOM_COOL_DOWN_SECONDS / _SECONDS_PER_RECORDING
                    else:
                        logger.warning(
                            "GLiClass CUDA graph recording failed; running eagerly from now on", exc_info=failure.cause
                        )
                        self._disabled = True
                    return failure.logits
                except Exception as exc:  # the eager forward that answers this call failed
                    if is_oom_error(exc):
                        # Release the graphs and let the worker's OOM recovery
                        # see the error.
                        self.clear()
                        raise
                    logger.warning("GLiClass CUDA graph recording failed; running eagerly from now on", exc_info=True)
                    self.clear()
                    self._disabled = True
                    return None
                self._graphs[key] = entry
                self._device_bytes += entry.device_bytes
                del entry  # the cache holds the only reference, so dropping it frees the graph
                while len(self._graphs) > self._max_graphs:
                    self._graphs.popitem(last=False)
                self._drop_unused_tables()
                held = self._device_bytes + self._tensor_bytes()
                if held > self._memory_budget(input_ids.device):
                    logger.info(
                        "GLiClass CUDA graphs took %d MB, over their budget; dropping them to record again",
                        held // 2**20,
                    )
                    self.clear()
                    # With every graph gone, their pool can go back to the device.
                    torch.cuda.empty_cache()
                return logits
            finally:
                _RECORDING_LOCK.release()

    def clear(self) -> None:
        """Drop every graph and its memory."""
        with self._lock:
            self._graphs.clear()
            self._sightings.clear()
            self._relative_pos.clear()
            self._device_bytes = 0

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def disabled(self) -> bool:
        return self._disabled

    # -- recording and replay ---------------------------------------------

    @staticmethod
    def _has_headroom(device: torch.device) -> bool:
        """Whether enough device memory is free to record without starving other models."""
        free, total = torch.cuda.mem_get_info(device)
        return free >= _RECORDING_HEADROOM * total

    @staticmethod
    def _memory_budget(device: torch.device) -> int:
        """Device memory this runner's recordings may take before its graphs are dropped."""
        return int(_MEMORY_BUDGET_SHARE * torch.cuda.mem_get_info(device)[1])

    @staticmethod
    def _free_memory(device: torch.device) -> int:
        return torch.cuda.mem_get_info(device)[0]

    def _tensor_bytes(self) -> int:
        """Device memory of the tensors the graphs keep: tables, static inputs and outputs."""
        tables = sum(table.nbytes for table in self._relative_pos.values())
        buffers = sum(
            sum(tensor.nbytes for tensor in entry.inputs.values()) + entry.output.nbytes
            for entry in self._graphs.values()
        )
        return tables + buffers

    def _drop_unused_tables(self) -> None:
        lengths = {length for _, length, _ in self._graphs}
        for length in [length for length in self._relative_pos if length not in lengths]:
            del self._relative_pos[length]

    def _static_inputs(self, key: Key, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch, length, _ = key
        static: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            buffer = torch.full((batch, length), self._pad_values[name], dtype=value.dtype, device=value.device)
            buffer[:, : value.shape[1]].copy_(value)
            static[name] = buffer
        return static

    def _record(self, key: Key, inputs: dict[str, torch.Tensor]) -> tuple[_Graph, torch.Tensor]:
        """Record the graph for ``key``; return it and this forward's logits.

        This call is answered by an eager forward on the current stream, as
        eager execution answers it; a replay of the new graph returns the same
        values. Its activations go back to the cache that eager forwards
        reuse. Recording runs on a stream of the runner's own, whose first use
        creates per-stream state (the cuBLAS handle and workspace) that cannot
        be created while recording: the first recording warms that stream up
        with one row of its inputs, the only memory left cached on it.

        Raises:
            _RecordingError: With the eager logits, when anything after the
                eager forward fails: the table, the warm-up or the recording.
        """
        classes = key[2]
        static = self._static_inputs(key, inputs)
        device = static["input_ids"].device
        current = torch.cuda.current_stream(device)
        with torch.inference_mode():
            logits = self._model(**static, max_num_classes=classes).logits.clone()
        try:
            return self._capture(key, static, current), logits
        except Exception as exc:
            raise _RecordingError(exc, logits) from exc

    def _capture(self, key: Key, static: dict[str, torch.Tensor], current: torch.cuda.Stream) -> _Graph:
        _, padded_length, classes = key
        device = static["input_ids"].device
        relative_pos = self._relative_pos.get(padded_length)
        if relative_pos is None:
            with torch.inference_mode():
                # Built eagerly, outside the recording: its CPU-to-GPU copy cannot be recorded.
                relative_pos = self._build_relative_pos(torch.empty((1, padded_length, 1), device=device))
            if relative_pos is not None:
                self._relative_pos[padded_length] = relative_pos
        warm_up = self._stream is None
        stream = self._stream or torch.cuda.Stream(device=device)
        pool = next(iter(self._graphs.values())).graph.pool() if self._graphs else None
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        self._recording_relative_pos = relative_pos
        self._recording = True
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                if warm_up:
                    self._model(**{name: value[:1] for name, value in static.items()}, max_num_classes=classes)
                    # Kept only once warmed up, so a failed warm-up is tried again.
                    self._stream = stream
                free_before = self._free_memory(device)
                graph.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    output = self._model(**static, max_num_classes=classes).logits
                except BaseException:
                    # End the recording without hiding why it failed.
                    with contextlib.suppress(Exception):
                        graph.capture_end()
                    raise
                graph.capture_end()
                device_bytes = max(0, free_before - self._free_memory(device))
        finally:
            self._recording = False
            self._recording_relative_pos = None
            current.wait_stream(stream)
        return _Graph(graph=graph, inputs=static, output=output, device_bytes=device_bytes)

    def _replay(self, entry: _Graph, inputs: dict[str, torch.Tensor], length: int) -> torch.Tensor:
        for name, value in inputs.items():
            buffer = entry.inputs[name]
            buffer[:, :length].copy_(value)
            if length < buffer.shape[1]:
                buffer[:, length:].fill_(self._pad_values[name])
        entry.graph.replay()
        return entry.output.clone()


def segment_ids(input_ids: torch.Tensor, config: Any) -> torch.Tensor:
    """The segment ids of gliclass ``_create_segment_ids``, without host reads.

    gliclass marks the text from its first text token with 1 and, when a row
    has an example token, everything from ``argmin`` of the example mask with
    2 (``argmin`` finds the first position that is not an example token,
    usually 0). This reproduces that exactly with tensor operations.
    """
    example_mask = input_ids == config.example_token_index
    example_start = example_mask.int().argmin(dim=-1)
    has_example = example_mask.any(dim=-1)
    text_start = (input_ids == config.text_token_index).int().argmax(dim=-1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    text = (positions >= text_start[:, None]).to(input_ids.dtype)
    with_examples = torch.where(positions >= example_start[:, None], torch.full_like(text, 2), text)
    return torch.where(has_example[:, None], with_examples, text)
