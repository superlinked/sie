"""CUDA graphs for the GLiClass encoder.

A GLiClass forward on a DeBERTa encoder launches about a thousand small
kernels, so at small batch sizes the GPU waits on the CPU that launches them.
A CUDA graph records the launches of one input shape once and replays them
with a single call.

Graphs are an operator setting, fixed when the model loads
(``adapter_options.loadtime.cuda_graphs``); a request can only opt out.

What a graph holds. A graph records the encoder: the embeddings, the DeBERTa
layers and, where a model has them, segment embeddings and layer-wise
attention, up to the hidden states gliclass scores from. The scoring head
(label and text features, pooling, projections and scorer) runs eagerly on
the replayed hidden states, with the forward's own number of label slots. It
is a few dozen small kernels, launched while the GPU still runs the graph. So
one graph serves every label count: graphs are recorded per (batch size,
sequence length) only.

A graph holds at most 2,048 tokens (batch times padded length), or 1,024 for
encoders wider than 768 (DeBERTa-v3-large); larger forwards run eagerly. Past
about that many tokens the GPU, not kernel launches, bounds a forward: per
layer, a forward's GPU time grows with the square of the hidden size while its
kernel launches do not. On an L4, a gliclass-large-v1.0 forward stops gaining from
its graph at about 1,000 tokens, a gliclass-base-v1.0 forward at about 2,000.

Two modes choose the shape:

- ``exact`` records the forward's own batch size and length. Replay launches
  the kernels eager execution launches, so scores are bit-identical to eager.
  A shape is recorded the second time it is seen, so shapes seen once cost
  nothing. The shapes are unbounded; a runner keeps the 64 most recently used
  graphs.
- ``bucketed`` pads the length up to a bucket (a multiple of 32 tokens at a
  512-token window, 64 at 1,024) and the batch up to a power of two, or to the
  largest batch the token bound allows at that length. Padded positions are
  masked, and padded rows are dropped before scoring. That makes the set of
  shapes small and fixed (52 for gliclass-large-v1.0, 71 for
  gliclass-base-v1.0, 33 for opir-multitask-large-v1.0), and the runner keeps
  a graph for every one, so after warm-up forwards replay whatever mix of
  batch sizes, lengths and label counts the traffic has. A longer sequence, or
  a larger batch, rounds fp16 sums differently, as when requests are batched
  together: up to 0.006 in probability on gliclass-large-v1.0 and 0.023 on
  gliclass-multilang-mini in our tests. A forward whose batch is already a
  bucket scores exactly as it did when graphs also covered the head.

While a graph is being recorded, PyTorch's caching allocator will not free
cached blocks to satisfy another allocation, so another model on the same GPU
can hit an out-of-memory error it would otherwise have avoided. Recording is
therefore kept rare and short:

- at most one recording at a time in the process, across all models; a
  forward that finds another recording in progress runs eagerly;
- no recording while less than a tenth of the device's memory is free;
- a runner has a budget of 16 recordings that refills at one every 2
  seconds: it records at most 16 graphs in quick succession, one after
  another, then about one every 2 seconds.

The forward that records a graph is answered by its first replay.

A runner's graphs hold device memory the server does not attribute to the
model: their shared pool, the driver's copy of each recorded graph (about
7 MB for a DeBERTa-large encoder), and the relative-position tables and
buffers they read. The graphs of a runner write their hidden states into one
shared buffer, so the pool holds only what a forward needs while it runs. The
runner adds up the device memory each recording takes, plus those tensors,
against 4% of the device's memory. On an L4, all the ``bucketed`` shapes of
each model that ships with graphs take 630 to 720 MB of that 900 MB. Where
they do not fit (a smaller GPU), recording stops at the budget and the graphs
already recorded keep replaying: dropping graphs to record others would only
record the same fixed shapes again. In ``exact`` mode, whose shapes are
unbounded, the runner past the budget drops every graph, returns the memory to
the device and starts again.

Recording needs a forward that never waits on the host. Two parts of the
gliclass DeBERTa forward do: the relative-position table copies a CPU scalar
to the GPU, and segment ids (instruct and Opir multitask models) read each
row's positions with ``.item()``. While recording, the runner hands the
encoder a table precomputed for the length, shared by the graphs of that
length, and computes segment ids with tensor operations; both give the same
integers.

Replays never overlap (a lock serializes them), and each replay's hidden
states are scored before the next replay. Only the recording thread's forward
is diverted while a graph records; the diversion is per thread, so an eager
forward of the same model on another thread runs unchanged. Anything
unsupported, larger than the token bound, or whose shape failed to record runs
eagerly. A recording or first replay that runs out of memory drops every
graph and pauses recording for a minute. A shape whose recording or first
replay fails for any reason but memory is not recorded again, and after three
such shapes the runner turns graphs off for the process. The scoring head is not part of that count: it
reads the forward's own inputs, so its errors fail that forward alone, as they
would eagerly. The runner
counts every forward it is offered (``CudaGraphStats``) and logs the counts
every ten minutes while it serves forwards.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from sie_server.adapters._cuda_graphs import (
    RECORDING_LOCK,
    CudaGraphStats,
    free_memory,
    has_headroom,
    memory_budget,
)
from sie_server.core.oom import is_oom_error

logger = logging.getLogger(__name__)

GraphMode = Literal["off", "exact", "bucketed"]
GRAPH_MODES: tuple[GraphMode, ...] = ("off", "exact", "bucketed")
# Recorded graphs an ``exact`` runner keeps, least recently used out. A
# ``bucketed`` runner keeps one for every shape it can record.
_MAX_GRAPHS = 64
# Shapes whose sightings ``exact`` mode counts.
_MAX_SIGHTINGS = 4096
# ``exact`` mode records a shape the time it is seen this many times.
_EXACT_RECORD_AT = 2
# Tokens (batch x padded length) a graph may hold. Past about that many tokens
# the GPU, not kernel launches, bounds a forward, so a graph saves little and
# would pin a larger memory pool. Encoders wider than this hidden size spend
# about twice the GPU time per token, so their graphs hold half as many.
_MAX_GRAPH_TOKENS = 2048
_WIDE_ENCODER_HIDDEN_SIZE = 768
# A runner's recording budget: this many recordings, refilled at one per this
# many seconds, however its forwards repeat. Recordings still run one at a time
# (``_RECORDING_LOCK``): recording costs a forward's launches, and while it runs
# the allocator cannot free cached memory for other models.
_RECORDING_BURST = 16
_SECONDS_PER_RECORDING = 2.0
# After a recording runs out of memory, the runner records nothing for this long.
_OOM_COOL_DOWN_SECONDS = 60.0
# One recording at a time in the process, across every model's runner and
# the ModernBERT varlen runner too (see ``sie_server.adapters._cuda_graphs``).
_RECORDING_LOCK = RECORDING_LOCK
# Shapes that may fail to record, for reasons other than memory, before the
# runner turns graphs off for the process.
_MAX_RECORDING_FAILURES = 3
# A runner serving forwards logs its counters this often.
_SUMMARY_SECONDS = 600.0
# Encoders whose forward records cleanly once the two host reads are replaced.
_ENCODERS = frozenset({"deberta-v2"})
# Poolings whose pooled vector ignores padded positions, so ``bucketed`` mode
# pads only the length.
_PADDING_SAFE_POOLINGS = frozenset({"first", "pass"})

# (batch size, sequence length)
Key = tuple[int, int]
# Why a forward the runner was offered ran eagerly.
EagerReason = Literal[
    "too_large",  # past the token bound or the model window
    "unsupported_inputs",  # inputs other than token ids, mask and token types
    "disabled",  # graphs turned off for the process after recording failures
    "unseen",  # ``exact`` mode: the shape's first sighting
    "failed_shape",  # the shape failed to record before
    "budget_full",  # ``bucketed`` mode: the memory budget holds no more graphs
    "recording_paused",  # out of recording credit, or cooling down after an out-of-memory error
    "no_headroom",  # too little free device memory to record
    "busy",  # another model's runner is recording
    "recording_failed",  # this forward's recording failed
]


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
    inner = getattr(model, "model", None)
    encoder = getattr(getattr(inner, "encoder_model", None), "encoder", None)
    if encoder is None or not hasattr(encoder, "get_rel_pos"):
        return "the encoder has no relative-position hook"
    if not callable(getattr(inner, "process_encoder_output", None)):
        return "the model has no scoring head to run apart from its encoder"
    return None


def bucket_width(max_length: int) -> int:
    """Sequence-length bucket for ``bucketed`` mode: 32 tokens at a 512 window, 64 at 1,024."""
    return max(16, max_length // 16)


def token_bound(hidden_size: int) -> int:
    """Tokens a graph of an encoder with this hidden size may hold."""
    return _MAX_GRAPH_TOKENS if hidden_size <= _WIDE_ENCODER_HIDDEN_SIZE else _MAX_GRAPH_TOKENS // 2


def batch_buckets(length: int, max_tokens: int) -> tuple[int, ...]:
    """Batch sizes ``bucketed`` graphs of this padded length hold: powers of two, and the largest that fits.

    A batch pads up to the next of these, so a batch past a power of two
    costs at most twice its rows, and one near the token bound pads only to
    the bound.
    """
    largest = max_tokens // length
    sizes = [1 << power for power in range(largest.bit_length())]
    if largest and sizes[-1] != largest:
        sizes.append(largest)
    return tuple(sizes)


def bucketed_shapes(max_length: int, max_tokens: int) -> frozenset[Key]:
    """Every shape a ``bucketed`` runner can record at this window."""
    width = bucket_width(max_length)
    lengths = {min(max_length, width * step) for step in range(1, math.ceil(max_length / width) + 1)}
    return frozenset((batch, length) for length in lengths for batch in batch_buckets(length, max_tokens))


class _RecordingOutOfMemoryError(Exception):
    """Recording a graph ran out of device memory."""


@dataclass
class _Graph:
    graph: Any
    inputs: dict[str, torch.Tensor]
    # Device memory the recording took: pool growth and the recorded graph.
    device_bytes: int = 0


class CudaGraphRunner:
    """Records and replays the encoder of one GLiClass model, per input shape."""

    def __init__(
        self,
        model: Any,
        *,
        pad_token_id: int,
        max_length: int,
        pad_token_type_id: int = 0,
        mode: GraphMode = "bucketed",
        name: str = "",
        max_graphs: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._name = name
        self._pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "token_type_ids": pad_token_type_id}
        self._max_length = max_length
        self._bucket = bucket_width(max_length)
        config = model.config
        self._hidden_size = int(config.encoder_config.hidden_size)
        embeddings = getattr(model.model.encoder_model, "get_input_embeddings", None)
        self._dtype = embeddings().weight.dtype if callable(embeddings) else torch.float16
        self._max_tokens = token_bound(self._hidden_size)
        self._padding_safe = config.pooling_strategy in _PADDING_SAFE_POOLINGS and not getattr(
            config, "use_lstm", False
        )
        # A bucketed runner whose lengths are bucketed records a fixed set of
        # shapes and keeps a graph for each; any other keeps the most recent.
        self._shapes = (
            bucketed_shapes(max_length, self._max_tokens) if mode == "bucketed" and self._padding_safe else None
        )
        if max_graphs is None:
            max_graphs = len(self._shapes) if self._shapes is not None else _MAX_GRAPHS
        self._max_graphs = max_graphs
        self._graphs: OrderedDict[Key, _Graph] = OrderedDict()
        self._sightings: OrderedDict[Key, int] = OrderedDict()
        self._failed: set[Key] = set()
        self._lock = threading.RLock()
        self._stream: torch.cuda.Stream | None = None
        self._disabled = False
        # Set when the memory budget holds no more of a bucketed runner's graphs.
        self._full = False
        self._clock = clock
        # Device memory recordings have taken since the graphs were last
        # dropped; evicting a graph does not give its share of the pool back.
        self._device_bytes = 0
        self._recording_credit = float(_RECORDING_BURST)
        self._credited_at = clock()
        # Relative-position tables of the recorded lengths; a graph reads its
        # length's table on every replay.
        self._relative_pos: dict[int, torch.Tensor] = {}
        # The hidden states every graph writes, sized for the token bound.
        self._hidden: torch.Tensor | None = None
        # Set only on the thread that is recording, while it records: the
        # hooks divert that thread's forward and leave every other thread's
        # alone.
        self._local = threading.local()
        self.stats = CudaGraphStats()
        self._summary_at = clock()
        self._summary_forwards = 0
        self._summary_replayed = 0
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
            if relative_pos is None and self._recording_here():
                return self._local.relative_pos
            return get_rel_pos(hidden_states, query_states, relative_pos)

        encoder.get_rel_pos = recording_get_rel_pos
        if getattr(self._model.config, "use_segment_embeddings", False):
            create_segment_ids = inner._create_segment_ids
            config = self._model.config

            def recording_segment_ids(input_ids: torch.Tensor) -> torch.Tensor:
                if not self._recording_here():
                    return create_segment_ids(input_ids)
                return segment_ids(input_ids, config)

            inner._create_segment_ids = recording_segment_ids

        # The head: everything the uni-encoder forward does after its encoder.
        # While recording, the forward stops there and returns the hidden states.
        head = inner.process_encoder_output
        self._head = head

        def recording_head(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_layer: torch.Tensor,
            labels: Any = None,
            max_num_classes: int | None = None,
        ) -> Any:
            if self._recording_here():
                return encoder_layer, None, None, None
            return head(input_ids, attention_mask, encoder_layer, labels, max_num_classes)

        inner.process_encoder_output = recording_head

    def _recording_here(self) -> bool:
        """Whether this thread is recording a graph of this runner's model."""
        return getattr(self._local, "recording", False)

    @contextlib.contextmanager
    def _recording_on_this_thread(self, relative_pos: torch.Tensor | None) -> Iterator[None]:
        """Divert this thread's forward to a recordable one while the block runs."""
        self._local.recording, self._local.relative_pos = True, relative_pos
        try:
            yield
        finally:
            self._local.recording, self._local.relative_pos = False, None

    # -- policy ------------------------------------------------------------

    def key(self, batch: int, length: int, mode: GraphMode) -> Key | None:
        """The graph a forward of this shape would replay; None when it runs eagerly."""
        if mode == "bucketed" and self._padding_safe:
            length = min(self._max_length, math.ceil(length / self._bucket) * self._bucket)
            batch = next((size for size in batch_buckets(length, self._max_tokens) if size >= batch), 0)
            if not batch:
                return None
        if batch * length > self._max_tokens or length > self._max_length:
            return None
        return (batch, length)

    def _why_not_record(self, key: Key, mode: GraphMode, device: torch.device) -> EagerReason | None:
        if key in self._failed:
            return "failed_shape"
        if mode == "exact":
            seen = self._sightings.pop(key, 0) + 1
            self._sightings[key] = seen
            while len(self._sightings) > _MAX_SIGHTINGS:
                self._sightings.popitem(last=False)
            if seen < _EXACT_RECORD_AT:
                return "unseen"
        if self._full:
            return "budget_full"
        if self._recording_credit < 1:
            return "recording_paused"
        if not self._has_headroom(device):
            return "no_headroom"
        return None

    # -- entry point -------------------------------------------------------

    def run(self, inputs: dict[str, torch.Tensor], max_num_classes: int | None, mode: GraphMode) -> torch.Tensor | None:
        """Logits for ``inputs`` from a graph, or None to run the forward eagerly."""
        if mode == "off":
            return None
        with self._lock:
            logits, reason = self._run(inputs, max_num_classes, mode)
            if reason is not None:
                self.stats.eager[reason] = self.stats.eager.get(reason, 0) + 1
            self._log_summary()
            return logits

    def _run(
        self, inputs: dict[str, torch.Tensor], max_num_classes: int | None, mode: GraphMode
    ) -> tuple[torch.Tensor | None, EagerReason | None]:
        if self._disabled:
            return None, "disabled"
        input_ids = inputs.get("input_ids")
        if (
            input_ids is None
            or input_ids.dim() != 2
            or "attention_mask" not in inputs  # the head reads it
            or set(inputs) - set(self._pad_values)
        ):
            return None, "unsupported_inputs"
        batch, length = input_ids.shape
        key = self.key(batch, length, mode)
        if key is None:
            return None, "too_large"
        now = self._clock()
        elapsed, self._credited_at = now - self._credited_at, now
        self._recording_credit = min(float(_RECORDING_BURST), self._recording_credit + elapsed / _SECONDS_PER_RECORDING)
        entry = self._graphs.get(key)
        if entry is not None:
            self._graphs.move_to_end(key)
            self.stats.replayed += 1
            return self._replay(key, entry, inputs, max_num_classes), None
        reason = self._why_not_record(key, mode, input_ids.device)
        if reason is not None:
            return None, reason
        if not _RECORDING_LOCK.acquire(blocking=False):
            return None, "busy"  # another model is recording
        try:
            self._recording_credit -= 1
            try:
                entry = self._record(key, inputs)
            except Exception as exc:  # noqa: BLE001 -- any failure to record runs the forward eagerly
                self._recording_failed(key, exc)
                return None, "recording_failed"
            # The graph answers the forward that recorded it, replayed before
            # the budget check below can drop it. Its first replay is part of
            # recording it, and the graph is not kept if it fails. Running out
            # of memory drops the graphs and pauses recording, as it does
            # while recording, and the error reaches the forward's
            # out-of-memory recovery; any other failure runs the forward
            # eagerly.
            try:
                hidden = self._replay_graph(key, entry, inputs)
            except Exception as exc:
                self._recording_failed(key, exc)
                if is_oom_error(exc):
                    raise
                return None, "recording_failed"
            self._keep(key, entry, input_ids.device)
            self.stats.recorded += 1
        finally:
            _RECORDING_LOCK.release()
        # The scoring head reads the forward's own inputs, so its errors are
        # the forward's, as in an eager forward: they fail this forward and
        # count nothing against the shape.
        return self._score(entry, hidden, max_num_classes), None

    def _keep(self, key: Key, entry: _Graph, device: torch.device) -> None:
        """Cache a new graph and hold the runner's graphs to their memory budget."""
        self._graphs[key] = entry
        self._device_bytes += entry.device_bytes
        del entry  # the cache holds the only reference, so dropping it frees the graph
        while len(self._graphs) > self._max_graphs:
            self._graphs.popitem(last=False)
        self._drop_unused_tables()
        held = self._device_bytes + self._tensor_bytes()
        if held <= self._memory_budget(device):
            return
        if self._shapes is not None:
            # A fixed set of shapes: dropping graphs to record others would
            # only record the same shapes again, so keep these.
            self._full = True
            logger.warning(
                "GLiClass CUDA graphs for %s took %d MB, their memory budget, with %d of %d shapes recorded; "
                "the other shapes run eagerly",
                self._name,
                held // 2**20,
                len(self._graphs),
                len(self._shapes),
            )
            return
        logger.info(
            "GLiClass CUDA graphs for %s took %d MB, over their budget; dropping them to record again",
            self._name,
            held // 2**20,
        )
        self.clear()
        self.stats.drops += 1
        # With every graph gone, their pool can go back to the device.
        torch.cuda.empty_cache()

    def _recording_failed(self, key: Key, exc: BaseException) -> None:
        if isinstance(exc, _RecordingOutOfMemoryError) or is_oom_error(exc):
            # Release the graphs; the forward runs eagerly, and recording pauses.
            self.clear()
            self.stats.drops += 1
            self._recording_credit = 1 - _OOM_COOL_DOWN_SECONDS / _SECONDS_PER_RECORDING
            logger.info(
                "GLiClass CUDA graph recording for %s ran out of memory; graphs dropped, recording paused for %d s",
                self._name,
                _OOM_COOL_DOWN_SECONDS,
            )
            return
        self._failed.add(key)
        self.stats.recording_failures += 1
        logger.warning(
            "GLiClass CUDA graph recording for %s failed for batch %d, length %d; that shape runs eagerly",
            self._name,
            key[0],
            key[1],
            exc_info=exc,
        )
        if len(self._failed) >= _MAX_RECORDING_FAILURES:
            self._disabled = True
            self.clear()
            logger.error(
                "GLiClass CUDA graphs for %s are off for this process after %d shapes failed to record; "
                "its forwards run eagerly",
                self._name,
                len(self._failed),
            )

    def clear(self) -> None:
        """Drop every graph and its memory."""
        with self._lock:
            self._graphs.clear()
            self._sightings.clear()
            self._relative_pos.clear()
            self._hidden = None
            self._device_bytes = 0
            self._full = False

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def shape_count(self) -> int | None:
        """How many shapes the runner can record; None when they are unbounded (``exact`` mode)."""
        return None if self._shapes is None else len(self._shapes)

    @property
    def held_bytes(self) -> int:
        """Device memory the runner counts against its budget."""
        return self._device_bytes + self._tensor_bytes()

    def _log_summary(self) -> None:
        now = self._clock()
        if now - self._summary_at < _SUMMARY_SECONDS:
            return
        stats = self.stats
        forwards = stats.forwards - self._summary_forwards
        replayed = stats.replayed - self._summary_replayed
        interval = now - self._summary_at
        self._summary_at, self._summary_forwards, self._summary_replayed = now, stats.forwards, stats.replayed
        if not forwards:
            return
        logger.info(
            "GLiClass CUDA graphs for %s: %d forwards in the last %d s, %.1f%% replayed; "
            "since load: %d replayed, %d recorded, eager %s, %d recording failures, %d drops; "
            "%d graphs, %d MB%s",
            self._name,
            forwards,
            round(interval),
            100.0 * replayed / forwards,
            stats.replayed,
            stats.recorded,
            dict(sorted(stats.eager.items())),
            stats.recording_failures,
            stats.drops,
            len(self._graphs),
            self.held_bytes // 2**20,
            ", off for this process" if self._disabled else "",
        )

    # -- recording and replay ---------------------------------------------

    @staticmethod
    def _has_headroom(device: torch.device) -> bool:
        """Whether enough device memory is free to record without starving other models."""
        return has_headroom(device)

    @staticmethod
    def _memory_budget(device: torch.device) -> int:
        """Device memory this runner's graphs may hold."""
        return memory_budget(device)

    @staticmethod
    def _free_memory(device: torch.device) -> int:
        return free_memory(device)

    def _tensor_bytes(self) -> int:
        """Device memory of the tensors the graphs keep: tables, static inputs and the hidden states."""
        tables = sum(table.nbytes for table in self._relative_pos.values())
        buffers = sum(sum(tensor.nbytes for tensor in entry.inputs.values()) for entry in self._graphs.values())
        hidden = self._hidden.nbytes if self._hidden is not None else 0
        return tables + buffers + hidden

    def _drop_unused_tables(self) -> None:
        lengths = {length for _, length in self._graphs}
        for length in [length for length in self._relative_pos if length not in lengths]:
            del self._relative_pos[length]

    def _hidden_states(self, key: Key, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """The shared hidden-state buffer, viewed as this shape's (batch, length, hidden) tensor."""
        batch, length = key
        if self._hidden is None:
            # Recorded graphs write here, so it lives as long as they do: only clear() drops it.
            self._hidden = torch.empty(self._max_tokens * self._hidden_size, dtype=dtype, device=device)
        elif self._hidden.dtype != dtype or self._hidden.device != device:
            msg = f"hidden states are {self._hidden.dtype} on {self._hidden.device}, not {dtype} on {device}"
            raise RuntimeError(msg)
        return self._hidden[: batch * length * self._hidden_size].view(batch, length, self._hidden_size)

    def _static_inputs(self, key: Key, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch, length = key
        return {
            name: torch.full((batch, length), self._pad_values[name], dtype=value.dtype, device=value.device)
            for name, value in inputs.items()
        }

    def _record(self, key: Key, inputs: dict[str, torch.Tensor]) -> _Graph:
        """Record the graph for ``key``.

        Recording runs on a stream of the runner's own, whose first use
        creates per-stream state (the cuBLAS handle and workspace) that cannot
        be created while recording: the first recording warms that stream up
        with one row of its inputs, the only memory left cached on it.
        Recording launches no kernels; the caller replays the graph to answer
        the forward.

        Raises:
            _RecordingOutOfMemoryError: When the warm-up or the recording runs out
                of device memory.
        """
        static = self._static_inputs(key, inputs)
        try:
            return self._capture(key, static)
        except Exception as exc:
            if is_oom_error(exc):
                raise _RecordingOutOfMemoryError(str(exc)) from exc
            raise

    def _capture(self, key: Key, static: dict[str, torch.Tensor]) -> _Graph:
        _, padded_length = key
        device = static["input_ids"].device
        current = torch.cuda.current_stream(device)
        relative_pos = self._relative_pos.get(padded_length)
        if relative_pos is None:
            with torch.inference_mode():
                # Built eagerly, outside the recording: its CPU-to-GPU copy cannot be recorded.
                relative_pos = self._build_relative_pos(torch.empty((1, padded_length, 1), device=device))
            if relative_pos is not None:
                self._relative_pos[padded_length] = relative_pos
        hidden = self._hidden_states(key, self._dtype, device)
        warm_up = self._stream is None
        stream = self._stream or torch.cuda.Stream(device=device)
        pool = next(iter(self._graphs.values())).graph.pool() if self._graphs else None
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        try:
            with self._recording_on_this_thread(relative_pos), torch.inference_mode(), torch.cuda.stream(stream):
                if warm_up:
                    self._model(**{name: value[:1] for name, value in static.items()})
                    # Kept only once warmed up, so a failed warm-up is tried again.
                    self._stream = stream
                free_before = self._free_memory(device)
                graph.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    encoded = self._model(**static).logits  # the head hook returns the hidden states
                    if encoded.shape != hidden.shape or encoded.dtype != hidden.dtype:
                        msg = f"expected hidden states of {tuple(hidden.shape)}, got {tuple(encoded.shape)}"
                        raise RuntimeError(msg)
                    hidden.copy_(encoded)
                    del encoded
                except BaseException:
                    # End the recording without hiding why it failed.
                    with contextlib.suppress(Exception):
                        graph.capture_end()
                    raise
                graph.capture_end()
                device_bytes = max(0, free_before - self._free_memory(device))
        finally:
            current.wait_stream(stream)
        return _Graph(graph=graph, inputs=static, device_bytes=device_bytes)

    def _replay(
        self, key: Key, entry: _Graph, inputs: dict[str, torch.Tensor], max_num_classes: int | None
    ) -> torch.Tensor:
        """Replay ``entry`` on ``inputs`` and score the replayed hidden states of the real rows."""
        return self._score(entry, self._replay_graph(key, entry, inputs), max_num_classes)

    def _replay_graph(self, key: Key, entry: _Graph, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Replay ``entry`` on ``inputs``; the hidden states of the real rows."""
        shared = self._hidden
        if shared is None:  # dropped only with every graph
            raise RuntimeError("GLiClass CUDA graph replayed without its hidden-state buffer")
        batch, length = next(iter(inputs.values())).shape
        for name, value in inputs.items():
            buffer = entry.inputs[name]
            pad = self._pad_values[name]
            buffer[:batch, :length].copy_(value)
            if length < buffer.shape[1]:
                buffer[:batch, length:].fill_(pad)
            if batch < buffer.shape[0]:
                buffer[batch:].fill_(pad)
        entry.graph.replay()
        return self._hidden_states(key, shared.dtype, shared.device)[:batch]

    def _score(self, entry: _Graph, hidden: torch.Tensor, max_num_classes: int | None) -> torch.Tensor:
        """The logits the scoring head gives the replayed hidden states of the real rows."""
        batch = hidden.shape[0]
        with torch.inference_mode():
            return self._head(
                entry.inputs["input_ids"][:batch], entry.inputs["attention_mask"][:batch], hidden, None, max_num_classes
            )[0]


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
