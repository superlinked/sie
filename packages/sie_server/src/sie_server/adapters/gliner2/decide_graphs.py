"""CUDA graphs for the GLiNER2.5-Decide encoder.

A Decide forward on its DeBERTa-v3-large encoder launches about a thousand
small GPU kernels from Python. Launching them takes the host 10 to 50 ms a
forward, whatever the batch size, and for a short message that is most of the
request: the GPU waits on the CPU. A CUDA graph records the launches of one
input shape once and replays them with a single call.

Graphs are an operator setting, fixed when the model loads
(``adapter_options.loadtime.cuda_graphs``: ``"off"`` or ``"bucketed"``).

What a graph holds. A graph records the encoder, from token ids to the last
hidden states, for one (batch size, padded length). The classification head
runs eagerly on the replayed hidden states: it reads each row's own label
markers, so one graph serves every label set and any number of labels.

Shapes. A call's rows pad to the length bucket of its longest row (a multiple
of 16 tokens up to 128, of 32 beyond) and to a batch size of 1, 2, 3, 4, 6, 8,
12, 16, 24 or 32 rows. A graph holds at most 2,048 tokens (batch times padded
length): 132 shapes at a 512-token window. Padded positions are masked, and a
padded row is a row of padding the caller never reads. A call that fits no
graph (more rows, or more tokens) runs eagerly, as it does without graphs.

Where graphs pay. A replay costs what the GPU needs for the padded shape and
nothing else. An eager forward costs at least what the host needs to launch its
kernels, whatever its size. So a graph is kept only if its replay takes no
longer than the host takes to launch one forward eagerly: such a graph cannot
lose to the eager forward it replaces, however few of its rows and tokens a
call fills. The runner times both when it records: the launch (one-row
forwards, three the first time and one with each later recording, keeping the
least ever seen) and the new graph's replay (the least of two). A graph that
takes longer is dropped and its shape runs eagerly, as does every shape with at
least its rows and its length, which the GPU cannot run faster. The rule
follows the machine: on a faster GPU, or a slower CPU, more shapes keep their
graph.

Both timings can only come out too long, when something else has the GPU or
the CPU at that moment: other engines warming up on the same GPU, another
model's forward. So neither verdict is final. A shape found slower is timed
again after a minute, then after two, doubling up to an hour, and is kept the
first time it pays. A graph kept against a launch time that later proves
shorter is dropped at its next replay. Without this, engines that warm up
together on one GPU each time their first graph while sharing it, find the
smallest shape slower than a launch, and run without graphs for good.

Measured on Modal in October 2026 (float16, rows of 64 tokens): one row takes
6.4 ms from a graph and 27.7 ms eagerly on an L4, 4.8 ms and 15.7 ms on an RTX
PRO 6000; eight rows take 15.6 ms and 27.2 ms on the L4, 5.2 ms and 15.2 ms on
the RTX PRO 6000. From about 1,000 tokens on that L4 and 3,000 on that RTX PRO
6000 the GPU bounds the forward and a graph saves nothing. A graph also keeps
the memory its recording used, which grows with its rows, so no graph holds
more than 2,048 tokens or 32 rows.

Scores. Replaying a graph launches the kernels an eager forward of the padded
shape launches, so a graph changes a score only as padding does: a row padded
to a longer length, or batched with other rows, rounds float16 sums differently.
That is the difference requests batched together already see without graphs.
On the 400 messages of the classify study no probability moved by more than
0.0014 between a graph and an eager forward of the same request (0.0019 between
an eager batch of eight and its requests one at a time), and no top label
changed.

While a graph is being recorded, PyTorch's caching allocator will not free
cached blocks to satisfy another allocation, so another model on the same GPU
can hit an out-of-memory error it would otherwise have avoided. Recording
follows the rules every runner in the process shares (``_cuda_graphs.py``):

- at most one recording at a time in the process, across all models; a forward
  that finds another recording in progress runs eagerly;
- no recording while less than a tenth of the device's memory is free;
- a runner has a budget of 16 recordings that refills at one every 2 seconds;
- a runner's graphs hold at most 4% of the device's memory: the pool the
  graphs share, the driver's copy of each graph (about 7 MB), their inputs,
  their relative-position tables and the hidden states they write. Past that,
  recording stops and the graphs already recorded keep replaying.

The forward that records a graph is answered by its first replay.

Recording needs a forward that never waits on the host. One part of the
DeBERTa forward does: building the relative-position table copies a CPU scalar
to the GPU. While recording, the runner hands the encoder a table built for the
length beforehand, shared by the graphs of that length; it holds the same
integers.

Replays never overlap: the caller holds ``lock`` from ``run`` until it has
queued the head that reads the hidden states, because every graph writes its
hidden states into one shared buffer. Only the recording thread's forward
is diverted while a graph records. A shape that fails to record, for any reason
but memory, runs eagerly from then on, and after three such shapes the runner
turns graphs off for the process. A recording or first replay that runs out of
memory drops every graph and pauses recording for a minute. The runner counts
every forward it is offered (``CudaGraphStats``) and logs the counts every ten
minutes while it serves forwards.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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

GraphMode = Literal["off", "bucketed"]
GRAPH_MODES: tuple[GraphMode, ...] = ("off", "bucketed")
# Tokens (batch x padded length) a graph may hold. Past about that many the GPU,
# not kernel launches, bounds a forward (see the module docstring).
MAX_GRAPH_TOKENS = 2048
# Batch sizes graphs are recorded for. A call pads up to the next of them, so
# under a third of a padded batch is padding (17 rows in a batch of 24).
_BATCH_SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
# Length buckets: this wide up to ``_NARROW_UNTIL`` tokens, ``_WIDE_BUCKET`` beyond.
_NARROW_BUCKET = 16
_NARROW_UNTIL = 128
_WIDE_BUCKET = 32
# A runner's recording budget: this many recordings, refilled at one per this many seconds.
_RECORDING_BURST = 16
_SECONDS_PER_RECORDING = 2.0
# A shape whose replay took longer than a launch is timed again after this long, then after twice
# as long each time it is found slower again, up to the cap.
_SLOW_RETRY_SECONDS = 60.0
_SLOW_RETRY_MAX_SECONDS = 3600.0
# After a recording runs out of memory, the runner records nothing for this long.
_OOM_COOL_DOWN_SECONDS = 60.0
# Shapes that may fail to record, for reasons other than memory, before the runner turns graphs off.
_MAX_RECORDING_FAILURES = 3
# A runner serving forwards logs its counters this often.
_SUMMARY_SECONDS = 600.0
_ENCODERS = frozenset({"deberta-v2"})

# (batch size, padded length)
Key = tuple[int, int]
# Why a forward the runner was offered ran eagerly.
EagerReason = Literal[
    "disabled",  # graphs turned off for the process after recording failures
    "failed_shape",  # the shape failed to record before
    "not_faster",  # the shape's replay takes longer than the host takes to launch a forward
    "budget_full",  # the memory budget holds no more graphs
    "recording_paused",  # out of recording credit, or cooling down after an out-of-memory error
    "no_headroom",  # too little free device memory to record
    "busy",  # another model's runner is recording
    "recording_failed",  # this forward's recording failed
]


def bucket_length(length: int, window: int) -> int:
    """The padded length of a row of ``length`` tokens."""
    width = _NARROW_BUCKET if length <= _NARROW_UNTIL else _WIDE_BUCKET
    return min(window, -(-length // width) * width)


def batch_sizes(length: int, max_tokens: int = MAX_GRAPH_TOKENS) -> tuple[int, ...]:
    """The batch sizes graphs of this padded length are recorded for."""
    return tuple(size for size in _BATCH_SIZES if size * length <= max_tokens)


def shapes(window: int, max_tokens: int = MAX_GRAPH_TOKENS) -> frozenset[Key]:
    """Every shape a runner can record at this window."""
    lengths = {bucket_length(length, window) for length in range(1, window + 1)}
    return frozenset((batch, length) for length in lengths for batch in batch_sizes(length, max_tokens))


def graph_shape(rows: int, longest: int, window: int, max_tokens: int = MAX_GRAPH_TOKENS) -> Key | None:
    """The graph a call of ``rows`` rows replays, the longest of ``longest`` tokens; None when it runs eagerly."""
    if rows < 1 or not 0 < longest <= window:
        return None
    length = bucket_length(longest, window)
    batch = next((size for size in batch_sizes(length, max_tokens) if size >= rows), None)
    return None if batch is None else (batch, length)


def unsupported_reason(
    encoder: Any, device: str | torch.device, window: int, max_tokens: int = MAX_GRAPH_TOKENS
) -> str | None:
    """Why ``encoder`` cannot use CUDA graphs of at most ``max_tokens`` on ``device``; None when it can."""
    if not str(device).startswith("cuda"):
        return "CUDA graphs need a CUDA device"
    encoder_type = getattr(getattr(encoder, "config", None), "model_type", None)
    if encoder_type not in _ENCODERS:
        return f"the {encoder_type} encoder is not supported"
    if not hasattr(getattr(encoder, "encoder", None), "get_rel_pos"):
        return "the encoder has no relative-position hook"
    if not batch_sizes(bucket_length(window, window), max_tokens):
        return f"a {window}-token row does not fit a graph"
    return None


class _RecordingOutOfMemoryError(Exception):
    """Recording a graph ran out of device memory."""


@dataclass
class _Graph:
    graph: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # Device memory the recording took: pool growth and the recorded graph.
    device_bytes: int = 0
    # Milliseconds the GPU took to replay it when it was recorded.
    replay_ms: float = 0.0


class DecideGraphRunner:
    """Records and replays the encoder of one GLiNER2.5-Decide model, per input shape."""

    def __init__(
        self,
        encoder: Any,
        *,
        window: int,
        pad_token_id: int = 0,
        name: str = "",
        max_tokens: int = MAX_GRAPH_TOKENS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._encoder = encoder
        self._name = name
        self._window = window
        self._pad_token_id = pad_token_id
        self._max_tokens = max_tokens
        self._hidden_size = int(encoder.config.hidden_size)
        self._shapes = shapes(window, max_tokens)
        self._graphs: dict[Key, _Graph] = {}
        self._failed: set[Key] = set()
        # Milliseconds this host takes to launch one eager forward: the least of every timing so far.
        self._launch_ms: float | None = None
        # Shapes whose replay took longer than that, each with the time it may be timed again and
        # how long it waits after the next such verdict. Until then they run eagerly.
        self._slow: dict[Key, tuple[float, float]] = {}
        # Held by the caller from ``run`` until it has queued the head (see the module docstring).
        self.lock = threading.RLock()
        self._stream: torch.cuda.Stream | None = None
        self._disabled = False
        # Set when the memory budget holds no more graphs.
        self._full = False
        self._clock = clock
        self._device_bytes = 0
        self._recording_credit = float(_RECORDING_BURST)
        self._credited_at = clock()
        # Relative-position tables of the recorded lengths; a graph reads its length's table on every replay.
        self._relative_pos: dict[int, torch.Tensor] = {}
        # The hidden states every graph writes, sized for the largest graph.
        self._hidden: torch.Tensor | None = None
        # Set only on the thread that is recording, while it records.
        self._local = threading.local()
        self.stats = CudaGraphStats()
        self._summary_at = clock()
        self._summary_forwards = 0
        self._summary_replayed = 0
        self._install_hook()

    # -- hook --------------------------------------------------------------

    def _install_hook(self) -> None:
        layers = self._encoder.encoder
        get_rel_pos = layers.get_rel_pos
        self._build_relative_pos = get_rel_pos

        def recording_get_rel_pos(
            hidden_states: torch.Tensor, query_states: Any = None, relative_pos: Any = None
        ) -> Any:
            if relative_pos is None and getattr(self._local, "recording", False):
                return self._local.relative_pos
            return get_rel_pos(hidden_states, query_states, relative_pos)

        layers.get_rel_pos = recording_get_rel_pos

    @contextlib.contextmanager
    def _recording_on_this_thread(self, relative_pos: torch.Tensor | None) -> Iterator[None]:
        """Hand this thread's forward the precomputed table while the block runs."""
        self._local.recording, self._local.relative_pos = True, relative_pos
        try:
            yield
        finally:
            self._local.recording, self._local.relative_pos = False, None

    # -- policy ------------------------------------------------------------

    def key(self, rows: int, longest: int) -> Key | None:
        """The graph a call of ``rows`` rows replays; None when it runs eagerly (see ``graph_shape``)."""
        return graph_shape(rows, longest, self._window, self._max_tokens)

    def _why_not_record(self, key: Key, device: torch.device) -> EagerReason | None:
        if key in self._failed:
            return "failed_shape"
        if self._presumed_slow(key):
            return "not_faster"  # no faster than a shape found slower than a launch
        if self._full:
            return "budget_full"
        if self._recording_credit < 1:
            return "recording_paused"
        if not self._has_headroom(device):
            return "no_headroom"
        return None

    def _presumed_slow(self, key: Key) -> bool:
        """Whether a shape with at most ``key``'s rows and length is, for now, found slower than a launch."""
        now = self._clock()
        return any(
            batch <= key[0] and length <= key[1] and now < again for (batch, length), (again, _) in self._slow.items()
        )

    def _found_slow(self, key: Key) -> None:
        """Run ``key`` eagerly until it may be timed again; the wait doubles with every such verdict."""
        wait = self._slow.get(key, (0.0, _SLOW_RETRY_SECONDS))[1]
        self._slow[key] = (self._clock() + wait, min(2 * wait, _SLOW_RETRY_MAX_SECONDS))

    # -- entry point -------------------------------------------------------

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, key: Key) -> torch.Tensor | None:
        """Hidden states of the rows of ``input_ids`` from the graph ``key``, or None to run them eagerly.

        ``input_ids`` holds at most ``key``'s rows and length. The result is a
        view of the buffer every graph writes: read it before the next ``run``.
        """
        with self.lock:
            hidden, reason = self._run(input_ids, attention_mask, key)
            if reason is not None:
                self.stats.eager[reason] = self.stats.eager.get(reason, 0) + 1
            self._log_summary()
            return hidden

    def _run(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, key: Key
    ) -> tuple[torch.Tensor | None, EagerReason | None]:
        if self._disabled:
            return None, "disabled"
        now = self._clock()
        elapsed, self._credited_at = now - self._credited_at, now
        self._recording_credit = min(float(_RECORDING_BURST), self._recording_credit + elapsed / _SECONDS_PER_RECORDING)
        entry = self._graphs.get(key)
        if entry is not None and self._launch_ms is not None and entry.replay_ms > self._launch_ms:
            # Kept against a launch time that has since proved shorter.
            del self._graphs[key]
            self._found_slow(key)
            return None, "not_faster"
        if entry is not None:
            self.stats.replayed += 1
            return self._replay(key, entry, input_ids, attention_mask), None
        reason = self._why_not_record(key, input_ids.device)
        if reason is not None:
            return None, reason
        if not RECORDING_LOCK.acquire(blocking=False):
            return None, "busy"  # another model is recording
        try:
            self._recording_credit -= 1
            device = input_ids.device
            free_before = self._free_memory(device)
            try:
                entry = self._record(key, device)
                pays = self._pays(key, entry)
            except Exception as exc:  # noqa: BLE001 -- any failure to record runs the forward eagerly
                self._recording_failed(key, exc)
                return None, "recording_failed"
            if not pays:
                # The graph goes; what its recording added to the pool the graphs share stays.
                del entry
                self._device_bytes += max(0, free_before - self._free_memory(device))
                return None, "not_faster"
            # The graph answers the forward that recorded it. Its first replay
            # is part of recording it: the graph is not kept if it fails.
            # Running out of memory reaches the forward's out-of-memory
            # recovery; any other failure runs the forward eagerly.
            try:
                hidden = self._replay(key, entry, input_ids, attention_mask)
            except Exception as exc:
                self._recording_failed(key, exc)
                if is_oom_error(exc):
                    raise
                return None, "recording_failed"
            self._keep(key, entry, input_ids.device)
            self.stats.recorded += 1
        finally:
            RECORDING_LOCK.release()
        return hidden, None

    def _pays(self, key: Key, entry: _Graph) -> bool:
        """Whether a new graph's replay takes no longer than the host takes to launch a forward eagerly.

        An eager forward takes at least its launches, so such a graph cannot
        lose to the eager forward it replaces. A graph that takes longer is not
        kept, and its shape runs eagerly until it is timed again.
        """
        first = self._launch_ms is None
        launch_ms = self._time_launch(key, entry, forwards=3 if first else 1)
        self._launch_ms = launch_ms if self._launch_ms is None else min(self._launch_ms, launch_ms)
        if first:
            logger.info(
                "GLiNER2.5-Decide CUDA graphs for %s: this host launches a forward in %.1f ms; a shape keeps its "
                "graph while the replay takes no longer",
                self._name,
                self._launch_ms,
            )
        entry.replay_ms = self._time_replay(entry)
        if entry.replay_ms <= self._launch_ms:
            self._slow.pop(key, None)
            return True
        self._found_slow(key)
        logger.debug(
            "GLiNER2.5-Decide CUDA graph for %s, batch %d, length %d, replays in %.1f ms, longer than a launch "
            "(%.1f ms); that shape runs eagerly until it is timed again",
            self._name,
            key[0],
            key[1],
            entry.replay_ms,
            self._launch_ms,
        )
        return False

    def _keep(self, key: Key, entry: _Graph, device: torch.device) -> None:
        """Cache a new graph and stop recording once the graphs take their memory budget."""
        self._graphs[key] = entry
        self._device_bytes += entry.device_bytes
        held = self.held_bytes
        if held <= self._memory_budget(device):
            return
        # A fixed set of shapes: dropping graphs to record others would only
        # record the same shapes again, so keep these.
        self._full = True
        logger.warning(
            "GLiNER2.5-Decide CUDA graphs for %s took %d MB, their memory budget, with %d of %d shapes recorded; "
            "the other shapes run eagerly",
            self._name,
            held // 2**20,
            len(self._graphs),
            len(self._shapes),
        )

    def _recording_failed(self, key: Key, exc: BaseException) -> None:
        if isinstance(exc, _RecordingOutOfMemoryError) or is_oom_error(exc):
            # Release the graphs; the forward runs eagerly, and recording pauses.
            self.clear()
            self.stats.drops += 1
            self._recording_credit = 1 - _OOM_COOL_DOWN_SECONDS / _SECONDS_PER_RECORDING
            logger.info(
                "GLiNER2.5-Decide CUDA graph recording for %s ran out of memory; graphs dropped, recording paused "
                "for %d s",
                self._name,
                _OOM_COOL_DOWN_SECONDS,
            )
            return
        self._failed.add(key)
        self.stats.recording_failures += 1
        logger.warning(
            "GLiNER2.5-Decide CUDA graph recording for %s failed for batch %d, length %d; that shape runs eagerly",
            self._name,
            key[0],
            key[1],
            exc_info=exc,
        )
        if len(self._failed) >= _MAX_RECORDING_FAILURES:
            self._disabled = True
            self.clear()
            logger.error(
                "GLiNER2.5-Decide CUDA graphs for %s are off for this process after %d shapes failed to record; "
                "its forwards run eagerly",
                self._name,
                len(self._failed),
            )

    def clear(self) -> None:
        """Drop every graph and its memory."""
        with self.lock:
            self._graphs.clear()
            self._relative_pos.clear()
            self._hidden = None
            self._device_bytes = 0
            self._full = False

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def shape_count(self) -> int:
        """How many shapes the runner can record."""
        return len(self._shapes)

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def launch_ms(self) -> float | None:
        """Milliseconds this host takes to launch one eager forward, the least timed; None before the first recording."""
        return self._launch_ms

    @property
    def held_bytes(self) -> int:
        """Device memory the runner counts against its budget."""
        tables = sum(table.nbytes for table in self._relative_pos.values())
        buffers = sum(entry.input_ids.nbytes + entry.attention_mask.nbytes for entry in self._graphs.values())
        hidden = self._hidden.nbytes if self._hidden is not None else 0
        return self._device_bytes + tables + buffers + hidden

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
            "GLiNER2.5-Decide CUDA graphs for %s: %d forwards in the last %d s, %.1f%% replayed; "
            "since load: %d replayed, %d recorded, eager %s, %d recording failures, %d drops; %d graphs, %d MB, "
            "%d shapes not faster than a launch%s",
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
            sum(now < again for again, _ in self._slow.values()),
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

    def _hidden_states(self, key: Key, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """The shared hidden-state buffer, viewed as this shape's (batch, length, hidden) tensor."""
        batch, length = key
        if self._hidden is None:
            # Recorded graphs write here, so it lives as long as they do: only clear() drops it.
            largest = max(batch * length for batch, length in self._shapes)
            self._hidden = torch.empty(largest * self._hidden_size, dtype=dtype, device=device)
        return self._hidden[: batch * length * self._hidden_size].view(batch, length, self._hidden_size)

    def _record(self, key: Key, device: torch.device) -> _Graph:
        """Record the graph for ``key``.

        Recording runs on a stream of the runner's own. A thread's first use
        of that stream creates state (the thread's cuBLAS handle and its
        workspace for the stream) that cannot be created while recording, so
        each thread's first recording warms the stream up with one row. The
        engine loads and warms a model up on one thread and serves it on
        another, so one warm-up for the runner would not do. Recording
        launches no kernels; the caller replays the graph to answer the
        forward.

        Raises:
            _RecordingOutOfMemoryError: When the warm-up or the recording runs out of device memory.
        """
        try:
            return self._capture(key, device)
        except Exception as exc:
            if is_oom_error(exc):
                raise _RecordingOutOfMemoryError(str(exc)) from exc
            raise

    def _capture(self, key: Key, device: torch.device) -> _Graph:
        batch, length = key
        current = torch.cuda.current_stream(device)
        with torch.inference_mode():
            input_ids = torch.full((batch, length), self._pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.ones((batch, length), dtype=torch.long, device=device)
            relative_pos = self._relative_pos.get(length)
            if relative_pos is None:
                # Built eagerly, outside the recording: its CPU-to-GPU copy cannot be recorded.
                relative_pos = self._build_relative_pos(torch.empty((1, length, 1), device=device))
                if relative_pos is not None:
                    self._relative_pos[length] = relative_pos
            dtype = self._encoder.get_input_embeddings().weight.dtype
            hidden = self._hidden_states(key, dtype, device)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        stream = self._stream
        warm_up = not getattr(self._local, "warmed", False)
        pool = next(iter(self._graphs.values())).graph.pool() if self._graphs else None
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        try:
            with self._recording_on_this_thread(relative_pos), torch.inference_mode(), torch.cuda.stream(stream):
                if warm_up:
                    self._encoder(input_ids=input_ids[:1], attention_mask=attention_mask[:1])
                    # Set only once warmed up, so a failed warm-up is tried again.
                    self._local.warmed = True
                free_before = self._free_memory(device)
                graph.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    encoded = self._encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
        return _Graph(graph=graph, input_ids=input_ids, attention_mask=attention_mask, device_bytes=device_bytes)

    def _time_launch(self, key: Key, entry: _Graph, forwards: int = 3) -> float:
        """Milliseconds the host takes to launch one eager forward: the least of ``forwards`` one-row forwards.

        An eager forward returns once its kernels are launched, without
        waiting for the GPU, and launches the same kernels whatever its size.
        The forward gets the precomputed table, as a recording does, so it
        does not wait on the host either.
        """
        device = entry.input_ids.device
        input_ids, attention_mask = entry.input_ids[:1], entry.attention_mask[:1]
        best = float("inf")
        with self._recording_on_this_thread(self._relative_pos.get(key[1])), torch.inference_mode():
            for _ in range(forwards):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                self._encoder(input_ids=input_ids, attention_mask=attention_mask)
                best = min(best, time.perf_counter() - start)
            torch.cuda.synchronize(device)
        return 1000.0 * best

    @staticmethod
    def _time_replay(entry: _Graph) -> float:
        """Milliseconds the GPU takes to run a graph: the least of two replays of its recorded inputs."""
        device = entry.input_ids.device
        best = float("inf")
        with torch.inference_mode():
            for _ in range(2):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                entry.graph.replay()
                torch.cuda.synchronize(device)
                best = min(best, time.perf_counter() - start)
        return 1000.0 * best

    def _replay(self, key: Key, entry: _Graph, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Replay ``entry`` on the rows of ``input_ids``; their hidden states."""
        shared = self._hidden
        if shared is None:  # dropped only with every graph
            raise RuntimeError("GLiNER2.5-Decide CUDA graph replayed without its hidden-state buffer")
        rows, length = input_ids.shape
        with torch.inference_mode():
            entry.input_ids[:rows, :length].copy_(input_ids)
            entry.attention_mask[:rows, :length].copy_(attention_mask)
            if length < key[1]:
                entry.input_ids[:rows, length:].fill_(self._pad_token_id)
                entry.attention_mask[:rows, length:].zero_()
            if rows < key[0]:
                # A padded row reads one padding token, so its forward is an ordinary one.
                entry.input_ids[rows:].fill_(self._pad_token_id)
                entry.attention_mask[rows:].zero_()
                entry.attention_mask[rows:, :1].fill_(1)
            entry.graph.replay()
        return self._hidden_states(key, shared.dtype, shared.device)[:rows]
