"""CUDA graphs for TopK-Embed-V1's small text batches.

A forward over a few short inputs (one query, a handful of queries) launches about
a thousand small kernels, so the GPU waits on the CPU that launches them. A CUDA
graph records the launches of one input shape once and replays them with a single
call.

Graphs are an operator setting, fixed when the model loads
(``adapter_options.loadtime.cuda_graphs``: ``off`` or ``bucketed``). They apply on
the packed CUDA path only (flash-linear-attention installed); images and anything
larger than a graph run packed and eagerly, as before.

What a graph holds. The embedding lookup and the backbone, in the ``Padded``
layout of ``packed.py``: every input is its own row, right-padded to the graph's
length. The Gated DeltaNet layers read each row left to right, so padding after
an input never reaches it, and attention masks the padded keys, so a replay gives
each input the hidden states the packed forward gives it, up to kernel rounding.
The head and normalisation run eagerly on the replayed hidden states.

Shapes (``bucketed``). A graph's length is the longest input rounded up: to a
multiple of 16 tokens up to 128, of 64 up to 512, of 256 past that. Its rows are
the batch rounded up to a power of two, or to the most rows that fit the token
bound at that length. A graph holds at most ``max_tokens`` tokens (rows times
length): by default 2,048 at a hidden size of 768 and proportionally fewer for
wider models (1,536 for the 0.8B model, 768 for the 2B). Past about that, the GPU,
not kernel launches, bounds a forward.

Recording follows the GLiClass runner's rules. One recording at a time in the
process (``sie_server.core.cuda_graph_recording``); none while less than a tenth
of the device's memory is free; at most 16 recordings at once, then one per 2
seconds; the graphs' memory is held to 4% of the device's, past which the shapes
not yet recorded run eagerly. A shape that fails to record runs eagerly from then
on, and after three such shapes graphs turn off for the process. Running out of
memory while recording drops every graph and pauses recording for a minute. The
forward that records a graph is answered by its first replay. Graphs share one
memory pool and one hidden-state buffer sized for the token bound; replays never
overlap (the adapter's forward lock), and each replay's hidden states are read
before the next.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from sie_server.adapters.topk_embed.packed import PackedTextModel, Padded
from sie_server.core.cuda_graph_recording import RECORDING_LOCK
from sie_server.core.oom import is_oom_error

logger = logging.getLogger(__name__)

GraphMode = Literal["off", "bucketed"]
GRAPH_MODES: tuple[GraphMode, ...] = ("off", "bucketed")
# A graph of a model with this hidden size holds this many tokens; wider models
# spend proportionally more GPU time per token, so their graphs hold fewer.
_BASE_GRAPH_TOKENS = 2048
_BASE_HIDDEN_SIZE = 768
# Length buckets: (up to this length, round up to a multiple of this).
_LENGTH_STEPS = ((128, 16), (512, 64))
_LONG_LENGTH_STEP = 256
# A runner may record this many graphs at once, then one per this many seconds.
_RECORDING_BURST = 16
_SECONDS_PER_RECORDING = 2.0
# Recording waits until at least this share of the device's memory is free.
_RECORDING_HEADROOM = 0.1
# After a recording runs out of memory, the runner records nothing for this long.
_OOM_COOL_DOWN_SECONDS = 60.0
# Device memory a runner's graphs may hold, as a share of the device's memory.
_MEMORY_BUDGET_SHARE = 0.04
# Shapes that may fail to record, for reasons other than memory, before graphs turn off.
_MAX_RECORDING_FAILURES = 3
# A runner serving forwards logs its counters this often.
_SUMMARY_SECONDS = 600.0

# (rows, padded length)
Key = tuple[int, int]
# Why a forward the runner was offered ran eagerly.
EagerReason = Literal[
    "too_large",  # past the token bound
    "disabled",  # graphs turned off for the process after recording failures
    "failed_shape",  # the shape failed to record before
    "budget_full",  # the memory budget holds no more graphs
    "recording_paused",  # out of recording credit, or cooling down after an out-of-memory error
    "no_headroom",  # too little free device memory to record
    "busy",  # another model's runner is recording
    "recording_failed",  # this forward's recording failed
]


def default_max_tokens(hidden_size: int) -> int:
    """Tokens a graph holds by default for a model with this hidden size (a multiple of 16)."""
    tokens = _BASE_GRAPH_TOKENS * _BASE_HIDDEN_SIZE // max(hidden_size, _BASE_HIDDEN_SIZE)
    return max(16, tokens // 16 * 16)


def bucket_length(length: int) -> int:
    """A graph's padded length for inputs up to ``length`` tokens."""
    for limit, step in _LENGTH_STEPS:
        if length <= limit:
            return max(step, math.ceil(length / step) * step)
    return math.ceil(length / _LONG_LENGTH_STEP) * _LONG_LENGTH_STEP


def row_buckets(length: int, max_tokens: int) -> tuple[int, ...]:
    """Row counts of the graphs of this padded length: powers of two, and the most that fit."""
    largest = max_tokens // length
    sizes = [1 << power for power in range(largest.bit_length())]
    if largest and sizes[-1] != largest:
        sizes.append(largest)
    return tuple(sizes)


def graph_key(rows: int, length: int, max_tokens: int) -> Key | None:
    """The graph a batch of ``rows`` inputs, the longest ``length`` tokens, replays; None when too large."""
    padded = bucket_length(length)
    size = next((size for size in row_buckets(padded, max_tokens) if size >= rows), None)
    return None if size is None else (size, padded)


def bucketed_shapes(max_tokens: int) -> frozenset[Key]:
    """Every shape a runner with this token bound can record."""
    lengths = {bucket_length(length) for length in range(1, max_tokens + 1)}
    return frozenset((rows, length) for length in lengths for rows in row_buckets(length, max_tokens))


@dataclass
class GraphStats:
    """Counters of the forwards a runner was offered since the model loaded.

    Each forward counts once: ``replayed`` (a recorded graph served it),
    ``recorded`` (it recorded a graph and was answered by the first replay), or
    under ``eager`` by why it ran eagerly. ``recording_failures`` counts shapes that
    failed to record for reasons other than memory; ``drops`` counts the times every
    graph was dropped.
    """

    replayed: int = 0
    recorded: int = 0
    eager: dict[str, int] = field(default_factory=dict)
    recording_failures: int = 0
    drops: int = 0

    @property
    def forwards(self) -> int:
        return self.replayed + self.recorded + sum(self.eager.values())


class _RecordingOutOfMemoryError(Exception):
    """Recording a graph ran out of device memory."""


@dataclass
class _Graph:
    graph: Any
    input_ids: torch.Tensor
    mask: torch.Tensor
    # Device memory the recording took: pool growth and the recorded graph.
    device_bytes: int = 0


class GraphRunner:
    """Records and replays TopK-Embed's embeddings and backbone per (rows, padded length)."""

    def __init__(
        self,
        text_model: PackedTextModel,
        embed: Any,
        *,
        pad_token_id: int,
        max_tokens: int,
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._text_model = text_model
        self._embed = embed
        self._pad_token_id = pad_token_id
        self._max_tokens = max_tokens
        self._name = name
        self._clock = clock
        self._hidden_size = int(embed.weight.shape[1])
        self._dtype = embed.weight.dtype
        self._graphs: dict[Key, _Graph] = {}
        self._failed: set[Key] = set()
        self._disabled = False
        # Set when the memory budget holds no more graphs.
        self._full = False
        self._device_bytes = 0
        self._recording_credit = float(_RECORDING_BURST)
        self._credited_at = clock()
        self._pool: Any = None
        self._stream: Any = None
        # The hidden states every graph writes, sized for the token bound.
        self._hidden: torch.Tensor | None = None
        self.stats = GraphStats()
        self._summary_at = clock()
        self._summary_forwards = 0
        self._summary_replayed = 0

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def held_bytes(self) -> int:
        """Device memory the runner counts against its budget."""
        buffers = sum(entry.input_ids.nbytes + entry.mask.nbytes for entry in self._graphs.values())
        hidden = self._hidden.nbytes if self._hidden is not None else 0
        return self._device_bytes + buffers + hidden

    # -- entry point -------------------------------------------------------

    def run(self, rows: list[torch.Tensor]) -> torch.Tensor | None:
        """Hidden states for token-id ``rows`` from a graph, or None to run them eagerly.

        The result is ``[len(rows), L, H]``: each input right-padded to the graph's
        length ``L``. It is a view of the runner's shared buffer, so read it before
        the next ``run``.
        """
        hidden, reason = self._run(rows)
        if reason is not None:
            self.stats.eager[reason] = self.stats.eager.get(reason, 0) + 1
        self._log_summary()
        return hidden

    def _run(self, rows: list[torch.Tensor]) -> tuple[torch.Tensor | None, EagerReason | None]:
        if self._disabled:
            return None, "disabled"
        key = graph_key(len(rows), max(len(row) for row in rows), self._max_tokens)
        if key is None:
            return None, "too_large"
        now = self._clock()
        elapsed, self._credited_at = now - self._credited_at, now
        self._recording_credit = min(float(_RECORDING_BURST), self._recording_credit + elapsed / _SECONDS_PER_RECORDING)
        entry = self._graphs.get(key)
        if entry is not None:
            self.stats.replayed += 1
            return self._replay(key, entry, rows), None
        reason = self._why_not_record(key)
        if reason is not None:
            return None, reason
        if not RECORDING_LOCK.acquire(blocking=False):
            return None, "busy"  # another model is recording
        try:
            self._recording_credit -= 1
            try:
                entry = self._record(key)
            except Exception as exc:  # noqa: BLE001 -- any failure to record runs the forward eagerly
                self._recording_failed(key, exc)
                return None, "recording_failed"
            # The graph answers the forward that recorded it. Its first replay is
            # part of recording it: a failure other than memory runs the forward
            # eagerly and the graph is not kept.
            try:
                hidden = self._replay(key, entry, rows)
            except Exception as exc:
                if is_oom_error(exc):
                    raise
                self._recording_failed(key, exc)
                return None, "recording_failed"
            self._keep(key, entry)
            self.stats.recorded += 1
        finally:
            RECORDING_LOCK.release()
        return hidden, None

    def _why_not_record(self, key: Key) -> EagerReason | None:
        if key in self._failed:
            return "failed_shape"
        if self._full:
            return "budget_full"
        if self._recording_credit < 1:
            return "recording_paused"
        if self._free_memory() < _RECORDING_HEADROOM * self._total_memory():
            return "no_headroom"
        return None

    def _keep(self, key: Key, entry: _Graph) -> None:
        """Cache a new graph; past the memory budget, record no more (the shapes are a fixed set)."""
        self._graphs[key] = entry
        self._device_bytes += entry.device_bytes
        held = self.held_bytes
        if held > _MEMORY_BUDGET_SHARE * self._total_memory():
            self._full = True
            logger.warning(
                "TopK-Embed CUDA graphs for %s took %d MB, their memory budget, with %d graphs recorded; "
                "the other shapes run eagerly",
                self._name,
                held // 2**20,
                len(self._graphs),
            )

    def _recording_failed(self, key: Key, exc: BaseException) -> None:
        if isinstance(exc, _RecordingOutOfMemoryError) or is_oom_error(exc):
            # Release the graphs; the forward runs eagerly, and recording pauses.
            self.clear()
            self.stats.drops += 1
            self._recording_credit = 1 - _OOM_COOL_DOWN_SECONDS / _SECONDS_PER_RECORDING
            logger.info(
                "TopK-Embed CUDA graph recording for %s ran out of memory; graphs dropped, recording paused for %d s",
                self._name,
                _OOM_COOL_DOWN_SECONDS,
            )
            return
        self._failed.add(key)
        self.stats.recording_failures += 1
        logger.warning(
            "TopK-Embed CUDA graph recording for %s failed for %d rows of %d tokens; that shape runs eagerly",
            self._name,
            key[0],
            key[1],
            exc_info=exc,
        )
        if len(self._failed) >= _MAX_RECORDING_FAILURES:
            self._disabled = True
            self.clear()
            logger.error(
                "TopK-Embed CUDA graphs for %s are off for this process after %d shapes failed to record",
                self._name,
                len(self._failed),
            )

    def clear(self) -> None:
        """Drop every graph and its memory."""
        self._graphs.clear()
        self._hidden = None
        self._pool = None
        self._device_bytes = 0
        self._full = False

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
            "TopK-Embed CUDA graphs for %s: %d forwards in the last %d s, %.1f%% replayed; since load: "
            "%d replayed, %d recorded, eager %s, %d recording failures, %d drops; %d graphs, %d MB%s",
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

    def _device(self) -> torch.device:
        return self._embed.weight.device

    def _free_memory(self) -> int:
        return torch.cuda.mem_get_info(self._device())[0]

    def _total_memory(self) -> int:
        return torch.cuda.mem_get_info(self._device())[1]

    def warm_up(self, step: int) -> None:
        """Run the recorded computation eagerly at each ``step`` of tokens a graph can hold.

        Its kernels are the padded variants of the packed path's, tuned on their own; the
        adapter's warm-up calls this so a first recording does not tune them.
        """
        device = self._embed.weight.device
        for tokens in range(step, self._max_tokens + step, step):
            ids = torch.full((1, min(tokens, self._max_tokens)), self._pad_token_id, dtype=torch.long, device=device)
            self._forward(ids, torch.ones_like(ids, dtype=torch.bool))

    def _forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The recorded computation: embeddings and backbone over right-padded rows."""
        rows, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).view(1, 1, -1).expand(3, rows, -1)
        return self._text_model(self._embed(input_ids), positions, Padded(mask))

    def _hidden_states(self, key: Key) -> torch.Tensor:
        """The shared hidden-state buffer, viewed as this shape's ``[rows, length, hidden]`` tensor."""
        rows, length = key
        if self._hidden is None:
            # Recorded graphs write here, so it lives as long as they do: only clear() drops it.
            self._hidden = torch.empty(self._max_tokens * self._hidden_size, dtype=self._dtype, device=self._device())
        return self._hidden[: rows * length * self._hidden_size].view(rows, length, self._hidden_size)

    def _record(self, key: Key) -> _Graph:
        """Record the graph for ``key``; raises ``_RecordingOutOfMemoryError`` when memory runs out."""
        try:
            return self._capture(key)
        except Exception as exc:
            if is_oom_error(exc):
                raise _RecordingOutOfMemoryError(str(exc)) from exc
            raise

    def _capture(self, key: Key) -> _Graph:
        """Record ``key`` on the runner's own stream, into the runner's memory pool.

        One eager forward of the shape runs on that stream first: Triton compiles and
        autotunes kernels for new shapes, and the stream's first use creates cuBLAS
        state, and neither can happen while recording. Recording launches no kernels;
        the caller replays the graph to answer the forward.
        """
        rows, length = key
        device = self._device()
        input_ids = torch.full((rows, length), self._pad_token_id, dtype=torch.long, device=device)
        mask = torch.ones((rows, length), dtype=torch.bool, device=device)
        hidden = self._hidden_states(key)
        current = torch.cuda.current_stream(device)
        stream = self._stream or torch.cuda.Stream(device=device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                self._forward(input_ids, mask)
                self._stream = stream
                free_before = self._free_memory()
                graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
                try:
                    hidden.copy_(self._forward(input_ids, mask))
                except BaseException:
                    # End the recording without hiding why it failed.
                    with contextlib.suppress(Exception):
                        graph.capture_end()
                    raise
                graph.capture_end()
                device_bytes = max(0, free_before - self._free_memory())
        finally:
            current.wait_stream(stream)
        return _Graph(graph=graph, input_ids=input_ids, mask=mask, device_bytes=device_bytes)

    def _replay(self, key: Key, entry: _Graph, rows: list[torch.Tensor]) -> torch.Tensor:
        """Replay ``entry`` on ``rows``; the hidden states of the real rows."""
        size, length = key
        input_ids = torch.full((size, length), self._pad_token_id, dtype=torch.long)
        mask = torch.zeros((size, length), dtype=torch.bool)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row
            mask[i, : len(row)] = True
        # Rows past the batch attend to their first position, so no attention row is empty.
        mask[len(rows) :, 0] = True
        entry.input_ids.copy_(input_ids)
        entry.mask.copy_(mask)
        entry.graph.replay()
        return self._hidden_states(key)[: len(rows)]
