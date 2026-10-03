"""CUDA graphs for the ModernBERT flash-attention varlen encoders.

The ModernBERT flash adapters (dense embeddings, late interaction and
cross-encoder reranking) pack a batch into one unpadded token stream and run
it through ``flash_attn_varlen_func``. A forward launches about 500 kernels
(some 23 per layer), and PyTorch spends roughly 20 microseconds of host time
on each. So below a few thousand packed tokens the GPU waits on the host that
launches them: on an L4, a one-query forward of a 22-layer, 768-wide encoder
keeps the GPU busy for 2 ms of its 12. A CUDA graph records those launches
once for one shape and replays them with one call.

Graphs are an operator setting, fixed when the model loads
(``adapter_options.loadtime.cuda_graphs: bucketed``).

What a graph holds. A graph records the encoder: the token embeddings and
their norm, the layers and the final norm, for one packed shape. The
adapter's own head (pooling, the late-interaction projection or the reranking
head) runs eagerly on the real rows of the graph's output, as it does after
an eager forward.

Shapes. A varlen forward has three shapes that vary: the packed token count,
the number of sequences and ``max_seqlen``. A graph fixes all three:

- the token count is padded up to a bucket: 64, 128 and the powers of two
  above it, with a midpoint (1.5 times) between each, up to a token bound;
- the sequences get a fixed number of slots, one per 16 tokens of the bucket
  (at least 8, at most 128). The real rows fill the first slots; the others
  are empty sequences. A forward with more rows than its bucket's slots uses
  a larger bucket;
- ``max_seqlen`` is 512 when every row fits in it, and the bucket otherwise.
  flash-attn launches a block per 128 query positions per slot per head
  whether or not a sequence fills it, so a graph for short rows launches
  fewer.

Padding tokens belong to no sequence: ``cu_seqlens`` ends at the real token
count. flash-attn never reads or writes their rows, and every other step of
the encoder (embedding lookup, norms, projections, the rotary embedding, the
MLP) works on each token alone, so no padding row reaches a real one, and
real rows attend exactly to their own tokens as eagerly. What changes is the
number of rows a matrix multiply sees, and with it the kernel cuBLAS picks
and the order it sums in: the same kind of change that batching requests
together makes. Outputs move by floating-point rounding (see the README).

The token bound depends on the encoder's width. Past about that many tokens
the GPU, not kernel launches, bounds a forward: a graph saves little, and
padding to the bucket can cost more than it saves. The bound is 2,048 tokens
up to 384 wide, 1,024 up to 768 and 512 wider (GPU time per token grows with
the square of the width, kernel launches do not). On an L4, a graph of a
768-wide, 22-layer encoder is 3.8-5.7x as fast as an eager forward up to 256
tokens, 1.4-1.6x at 1,024, and at 1,200 tokens padded to 1,536 it is slower
(0.9x); a 384-wide one still gains 1.25x at 2,048. Larger forwards run
eagerly.

Recording follows the GLiClass DeBERTa graphs (``gliclass/cuda_graphs.py``)
and shares their process-wide rules (``_cuda_graphs.py``): one recording at a
time in the process, none while less than a tenth of the device's memory is
free, a budget of 16 recordings per model that refills at one every 2 seconds
(16 in quick succession, one after another, then about one every 2 seconds),
and the graphs of one model within 4% of the device's memory. The set of shapes is
fixed, so when a model's graphs reach that budget the runner stops recording
and keeps replaying the graphs it has. A shape is recorded the first time a
forward needs it (the dense adapter's warm-up forward, at load, records the
smallest), and the new graph's first replay answers that forward. A
recording that runs out of memory drops every graph and pauses recording for
a minute. A shape whose recording or first replay fails for any other reason
is not recorded again, and after three such shapes the runner turns graphs
off for the process. The adapter's head is not part of that count: its
errors fail the forward alone, as they would eagerly.

Replays never overlap: a lock serializes them, and each replay's output is
read by the adapter's head before the next replay. The runner counts every
forward it is offered (``CudaGraphStats``) and logs the counts every ten
minutes while it serves forwards.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import numpy as np
import torch

from sie_server.adapters._cuda_graphs import (
    RECORDING_LOCK,
    CudaGraphStats,
    free_memory,
    has_headroom,
    memory_budget,
)
from sie_server.adapters._modernbert_flash import (
    modernbert_rope_cos_sin,
    modernbert_rope_theta,
    run_modernbert_flash_layers,
)
from sie_server.core.oom import is_oom_error

logger = logging.getLogger(__name__)

GraphMode = Literal["off", "bucketed"]
GRAPH_MODES: tuple[GraphMode, ...] = ("off", "bucketed")

# The smallest bucket, and the first to get a midpoint above it.
_MIN_TOKENS = 64
_FIRST_MIDPOINT = 128
# Sequence slots: one per this many tokens of the bucket, within these bounds.
_TOKENS_PER_SLOT = 16
_MIN_SLOTS = 8
_MAX_SLOTS = 128
# ``max_seqlen`` of a graph for rows that all fit in it.
_SHORT_ROWS = 512
# A runner's recording budget: this many recordings, refilled at one per this
# many seconds. Recordings still run one at a time (``RECORDING_LOCK``).
_RECORDING_BURST = 16
_SECONDS_PER_RECORDING = 2.0
# After a recording runs out of memory, the runner records nothing for this long.
_OOM_COOL_DOWN_SECONDS = 60.0
# Shapes that may fail to record, for reasons other than memory, before the
# runner turns graphs off for the process.
_MAX_RECORDING_FAILURES = 3
# A runner serving forwards logs its counters this often.
_SUMMARY_SECONDS = 600.0

# (padded token count, max_seqlen)
Key = tuple[int, int]
# Why a forward the runner was offered ran eagerly.
EagerReason = Literal[
    "empty",  # no tokens
    "too_long",  # a row longer than the model window
    "too_large",  # past the token bound, or more rows than any bucket has slots
    "disabled",  # graphs turned off for the process after recording failures
    "failed_shape",  # the shape failed to record before
    "budget_full",  # the memory budget holds no more graphs
    "recording_paused",  # out of recording credit, or cooling down after an out-of-memory error
    "no_headroom",  # too little free device memory to record
    "busy",  # another model's runner is recording
    "recording_failed",  # this forward's recording failed
]

# The encoder a graph records: (token ids, positions, cu_seqlens, max_seqlen,
# total tokens) -> hidden states ``[total tokens, hidden]``. Ids, positions
# and cu_seqlens are int32 device tensors.
EncodeFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int], torch.Tensor]
T = TypeVar("T")


def token_bound(hidden_size: int) -> int:
    """Packed tokens a graph of an encoder with this hidden size may hold."""
    if hidden_size <= 384:
        return 2048
    if hidden_size <= 768:
        return 1024
    return 512


def token_buckets(max_tokens: int) -> tuple[int, ...]:
    """Packed-token sizes graphs hold: 64, 128, 192, 256, 384, 512, 768, ... up to ``max_tokens``."""
    sizes: list[int] = []
    size = _MIN_TOKENS
    while size <= max_tokens:
        sizes.append(size)
        midpoint = size * 3 // 2
        if size >= _FIRST_MIDPOINT and midpoint <= max_tokens:
            sizes.append(midpoint)
        size *= 2
    return tuple(sizes)


def row_slots(tokens: int) -> int:
    """Sequence slots of a graph holding this many tokens."""
    return min(_MAX_SLOTS, max(_MIN_SLOTS, tokens // _TOKENS_PER_SLOT))


def seqlen_buckets(tokens: int, window: int) -> tuple[int, ...]:
    """The ``max_seqlen`` values graphs of this token bucket are recorded with."""
    longest = min(tokens, window)
    short = min(_SHORT_ROWS, longest)
    return (short,) if short == longest else (short, longest)


def bucketed_shapes(max_tokens: int, window: int) -> frozenset[Key]:
    """Every shape a runner can record."""
    return frozenset(
        (tokens, seqlen) for tokens in token_buckets(max_tokens) for seqlen in seqlen_buckets(tokens, window)
    )


def unsupported_reason(device: str | torch.device) -> str | None:
    """Why graphs cannot run on ``device``; None when they can."""
    if not str(device).startswith("cuda"):
        return "CUDA graphs need a CUDA device"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    return None


def parse_graph_mode(value: object, *, adapter: str) -> GraphMode:
    """The load-time ``cuda_graphs`` option; YAML reads an unquoted ``off`` as False."""
    if value is False:
        return "off"
    if isinstance(value, str) and value in GRAPH_MODES:
        return cast("GraphMode", value)
    msg = f"{adapter} cuda_graphs must be 'off' or 'bucketed', got {value!r}"
    raise ValueError(msg)


def graph_runner(
    build: Callable[[], tuple[EncodeFn, Sequence[torch.Tensor]]],
    *,
    mode: GraphMode,
    device: str,
    hidden_size: int,
    dtype: torch.dtype,
    window: int,
    pad_token_id: int | None,
    name: str,
) -> VarlenGraphRunner | None:
    """A runner for a loaded model; None when graphs are off or the device cannot run them.

    ``build`` returns the encoder a graph records and the tensors it reads
    (RoPE tables), counted against the runner's memory budget.
    """
    if mode == "off":
        return None
    reason = unsupported_reason(device)
    if reason is not None:
        logger.warning("ModernBERT cuda_graphs=%s does not apply to %s, which runs eagerly: %s", mode, name, reason)
        return None
    encode, static_tensors = build()
    return VarlenGraphRunner(
        encode,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
        window=window,
        pad_token_id=pad_token_id or 0,
        static_tensors=static_tensors,
        name=name,
    )


def modernbert_encoder(
    model: Any, *, window: int, dtype: torch.dtype, fused_rope: bool = False
) -> tuple[EncodeFn, list[torch.Tensor]]:
    """The encoder a graph records for a Hugging Face ``ModernBertModel`` on the shared layer stack.

    Token embeddings (and their norm), ``run_modernbert_flash_layers`` and the
    final norm: what the dense and late-interaction adapters run eagerly.
    Eagerly they compute each forward's RoPE ``cos``/``sin`` rows from its
    positions; here the rows are gathered from tables over the model window,
    computed the same way, so they hold the same values. ``fused_rope`` goes to
    the layer stack, so a graph records the rotation its model runs eagerly.

    Returns:
        The encoder, and the RoPE tables it reads (for the runner's memory budget).
    """
    config = model.config
    weight = model.embeddings.tok_embeddings.weight
    positions = torch.arange(window, device=weight.device)
    head_dim = config.hidden_size // config.num_attention_heads
    tables = {
        kind: modernbert_rope_cos_sin(
            positions,
            head_dim=head_dim,
            theta=modernbert_rope_theta(config, use_global=kind == "global"),
            dtype=dtype,
        )
        for kind in ("global", "local")
    }
    global_cos, global_sin = tables["global"]
    local_cos, local_sin = tables["local"]
    embeddings = model.embeddings

    def encode(
        input_ids: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, total: int
    ) -> torch.Tensor:
        hidden = embeddings.tok_embeddings(input_ids)
        if hasattr(embeddings, "norm"):
            hidden = embeddings.norm(hidden)
        if hasattr(embeddings, "drop"):
            hidden = embeddings.drop(hidden)
        hidden = run_modernbert_flash_layers(
            model,
            hidden,
            cu_seqlens,
            max_seqlen,
            total,
            global_cos[positions],
            global_sin[positions],
            local_cos[positions],
            local_sin[positions],
            fused_rope=fused_rope,
        )
        if hasattr(model, "final_norm"):
            hidden = model.final_norm(hidden)
        return hidden

    return encode, [global_cos, global_sin, local_cos, local_sin]


@dataclass
class PackedForward:
    """The real rows of a replayed forward, as device tensors the adapter's head reads.

    The tensors are views of the runner's buffers: read them before the
    runner is used again (the head runs while the runner holds its lock).
    """

    hidden: torch.Tensor  # [total tokens, hidden], after the final norm
    cu_seqlens: torch.Tensor  # [rows + 1], int32
    input_ids: torch.Tensor  # [total tokens], int32
    lengths: list[int]


class _RecordingOutOfMemoryError(Exception):
    """Recording a graph ran out of device memory."""


@dataclass
class _Graph:
    graph: Any
    # int32 [2 * tokens + slots + 1]: token ids, positions, cu_seqlens.
    inputs: torch.Tensor
    # Device memory the recording took: pool growth and the recorded graph.
    device_bytes: int = 0


class VarlenGraphRunner:
    """Records and replays the packed encoder of one model, per bucketed shape.

    Args:
        encode: The encoder a graph records (``EncodeFn``).
        hidden_size: Width of the encoder's output.
        dtype: dtype of the encoder's output.
        device: The model's CUDA device.
        window: The longest row the model serves (its RoPE tables' length).
        pad_token_id: Token id at padding positions (any id in the vocabulary).
        max_tokens: Token bound; defaults to ``token_bound(hidden_size)``.
        static_tensors: Device tensors the graphs read that the runner should
            count against its memory budget (RoPE tables).
        name: Model name for logs.
        clock: Monotonic clock (tests move it).
    """

    def __init__(
        self,
        encode: EncodeFn,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
        window: int,
        pad_token_id: int = 0,
        max_tokens: int | None = None,
        static_tensors: Sequence[torch.Tensor] = (),
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._encode = encode
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._device = torch.device(device)
        self._window = window
        self._pad_token_id = pad_token_id
        self._max_tokens = max_tokens if max_tokens is not None else token_bound(hidden_size)
        self._tokens = token_buckets(self._max_tokens)
        self._shapes = bucketed_shapes(self._max_tokens, window)
        self._static_bytes = sum(tensor.nbytes for tensor in static_tensors)
        self._name = name
        self._clock = clock
        self._graphs: OrderedDict[Key, _Graph] = OrderedDict()
        self._failed: set[Key] = set()
        self._lock = threading.RLock()
        self._stream: torch.cuda.Stream | None = None
        self._disabled = False
        # Set when the memory budget holds no more graphs.
        self._full = False
        # Device memory recordings have taken since the graphs were last dropped.
        self._device_bytes = 0
        self._recording_credit = float(_RECORDING_BURST)
        self._credited_at = clock()
        # The hidden states every graph writes, sized for the token bound.
        self._hidden: torch.Tensor | None = None
        self.stats = CudaGraphStats()
        self._summary_at = clock()
        self._summary_forwards = 0
        self._summary_replayed = 0

    # -- policy ------------------------------------------------------------

    def key(self, total: int, rows: int, longest: int) -> Key | None:
        """The graph a forward of this shape replays; None when it runs eagerly."""
        for tokens in self._tokens:
            if tokens >= total and row_slots(tokens) >= rows:
                seqlens = seqlen_buckets(tokens, self._window)
                return (tokens, seqlens[0] if longest <= seqlens[0] else seqlens[-1])
        return None

    def _why_not_record(self, key: Key) -> EagerReason | None:
        if key in self._failed:
            return "failed_shape"
        if self._full:
            return "budget_full"
        if self._recording_credit < 1:
            return "recording_paused"
        if not self._has_headroom(self._device):
            return "no_headroom"
        return None

    # -- entry point -------------------------------------------------------

    def run(
        self, token_ids: Sequence[int] | np.ndarray, lengths: Sequence[int], head: Callable[[PackedForward], T]
    ) -> T | None:
        """``head`` applied to the replayed encoder output, or None to run the forward eagerly.

        Args:
            token_ids: The packed token ids of the real rows, row after row.
            lengths: Tokens per row.
            head: The adapter's head. It must not return None, and must not
                keep the ``PackedForward`` tensors past its return.
        """
        with self._lock:
            try:
                result, reason = self._run(token_ids, lengths, head)
            except Exception as exc:
                # Graph memory is held until its graphs go: drop them so the
                # caller's out-of-memory recovery has that memory to work with.
                if is_oom_error(exc):
                    self.clear()
                    self.stats.drops += 1
                raise
            if reason is not None:
                self.stats.eager[reason] = self.stats.eager.get(reason, 0) + 1
            self._log_summary()
            return result

    def _run(
        self, token_ids: Sequence[int] | np.ndarray, lengths: Sequence[int], head: Callable[[PackedForward], T]
    ) -> tuple[T | None, EagerReason | None]:
        if self._disabled:
            return None, "disabled"
        total = int(sum(lengths))
        if not total:
            return None, "empty"
        longest = max(lengths)
        if longest > self._window:
            return None, "too_long"
        key = self.key(total, len(lengths), longest)
        if key is None:
            return None, "too_large"
        now = self._clock()
        elapsed, self._credited_at = now - self._credited_at, now
        self._recording_credit = min(float(_RECORDING_BURST), self._recording_credit + elapsed / _SECONDS_PER_RECORDING)
        entry = self._graphs.get(key)
        if entry is not None:
            self.stats.replayed += 1
            return head(self._replay(key, entry, token_ids, lengths)), None
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
            # The graph answers the forward that recorded it. Its first replay
            # is part of recording it: a failure other than memory runs the
            # forward eagerly and the graph is not kept.
            try:
                packed = self._replay(key, entry, token_ids, lengths)
            except Exception as exc:
                if is_oom_error(exc):
                    # Part of recording: pause as an out-of-memory recording does.
                    self._recording_credit = 1 - _OOM_COOL_DOWN_SECONDS / _SECONDS_PER_RECORDING
                    raise
                self._recording_failed(key, exc)
                return None, "recording_failed"
            self._keep(key, entry)
            self.stats.recorded += 1
        finally:
            RECORDING_LOCK.release()
        # The head reads the forward's own rows, so its errors are the
        # forward's, as eagerly: they fail this forward and count nothing
        # against the shape.
        return head(packed), None

    def _keep(self, key: Key, entry: _Graph) -> None:
        """Cache a new graph and hold the runner's graphs to their memory budget."""
        self._graphs[key] = entry
        self._device_bytes += entry.device_bytes
        held = self.held_bytes
        if held <= self._memory_budget(self._device):
            return
        # A fixed set of shapes: dropping graphs to record others would only
        # record the same shapes again, so keep these.
        self._full = True
        logger.warning(
            "ModernBERT CUDA graphs for %s took %d MB, their memory budget, with %d of %d shapes recorded; "
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
                "ModernBERT CUDA graph recording for %s ran out of memory; graphs dropped, recording paused for %d s",
                self._name,
                _OOM_COOL_DOWN_SECONDS,
            )
            return
        self._failed.add(key)
        self.stats.recording_failures += 1
        logger.warning(
            "ModernBERT CUDA graph recording for %s failed for %d tokens, max_seqlen %d; that shape runs eagerly",
            self._name,
            key[0],
            key[1],
            exc_info=exc,
        )
        if len(self._failed) >= _MAX_RECORDING_FAILURES:
            self._disabled = True
            self.clear()
            logger.error(
                "ModernBERT CUDA graphs for %s are off for this process after %d shapes failed to record; "
                "its forwards run eagerly",
                self._name,
                len(self._failed),
            )

    def clear(self) -> None:
        """Drop every graph and its memory."""
        with self._lock:
            self._graphs.clear()
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
    def shape_count(self) -> int:
        """How many shapes the runner can record."""
        return len(self._shapes)

    @property
    def held_bytes(self) -> int:
        """Device memory the runner counts against its budget."""
        inputs = sum(entry.inputs.nbytes for entry in self._graphs.values())
        hidden = self._hidden.nbytes if self._hidden is not None else 0
        return self._device_bytes + inputs + hidden + self._static_bytes

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
            "ModernBERT CUDA graphs for %s: %d forwards in the last %d s, %.1f%% replayed; "
            "since load: %d replayed, %d recorded, eager %s, %d recording failures, %d drops; "
            "%d of %d graphs, %d MB%s",
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
            len(self._shapes),
            self.held_bytes // 2**20,
            ", off for this process" if self._disabled else "",
        )

    # -- recording and replay ---------------------------------------------

    @staticmethod
    def _has_headroom(device: torch.device) -> bool:
        return has_headroom(device)

    @staticmethod
    def _memory_budget(device: torch.device) -> int:
        return memory_budget(device)

    @staticmethod
    def _free_memory(device: torch.device) -> int:
        return free_memory(device)

    def _hidden_states(self, tokens: int) -> torch.Tensor:
        """The shared output buffer, viewed as ``[tokens, hidden]``."""
        if self._hidden is None:
            # Recorded graphs write here, so it lives as long as they do: only clear() drops it.
            self._hidden = torch.empty(self._max_tokens * self._hidden_size, dtype=self._dtype, device=self._device)
        return self._hidden[: tokens * self._hidden_size].view(tokens, self._hidden_size)

    def _record(self, key: Key) -> _Graph:
        """Record the graph for ``key``.

        Recording runs on a stream of the runner's own, whose first use
        creates per-stream state (the cuBLAS handle and workspace, compiled
        Triton kernels) that cannot be created while recording: the first
        recording warms that stream up with a forward of the smallest shape,
        the only memory left cached on it. Recording launches no kernels; the
        caller replays the graph to answer the forward.

        Raises:
            _RecordingOutOfMemoryError: When the warm-up or the recording runs out
                of device memory.
        """
        try:
            return self._capture(key)
        except Exception as exc:
            if is_oom_error(exc):
                raise _RecordingOutOfMemoryError(str(exc)) from exc
            raise

    def _capture(self, key: Key) -> _Graph:
        tokens, seqlen = key
        slots = row_slots(tokens)
        # Every slot empty and every id and position 0: valid to warm up on.
        inputs = torch.zeros(2 * tokens + slots + 1, dtype=torch.int32, device=self._device)
        hidden = self._hidden_states(tokens)
        current = torch.cuda.current_stream(self._device)
        warm_up = self._stream is None
        stream = self._stream or torch.cuda.Stream(device=self._device)
        pool = next(iter(self._graphs.values())).graph.pool() if self._graphs else None
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                if warm_up:
                    small = torch.zeros(
                        2 * _MIN_TOKENS + row_slots(_MIN_TOKENS) + 1, dtype=torch.int32, device=self._device
                    )
                    self._forward(small, _MIN_TOKENS, _MIN_TOKENS)
                    # Kept only once warmed up, so a failed warm-up is tried again.
                    self._stream = stream
                free_before = self._free_memory(self._device)
                graph.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    encoded = self._forward(inputs, tokens, seqlen)
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
                device_bytes = max(0, free_before - self._free_memory(self._device))
        finally:
            current.wait_stream(stream)
        return _Graph(graph=graph, inputs=inputs, device_bytes=device_bytes)

    def _forward(self, inputs: torch.Tensor, tokens: int, seqlen: int) -> torch.Tensor:
        ids, positions, cu_seqlens = inputs[:tokens], inputs[tokens : 2 * tokens], inputs[2 * tokens :]
        return self._encode(ids, positions, cu_seqlens, seqlen, tokens)

    def _replay(
        self, key: Key, entry: _Graph, token_ids: Sequence[int] | np.ndarray, lengths: Sequence[int]
    ) -> PackedForward:
        """Replay ``entry`` on the real rows; their hidden states and layout."""
        if self._hidden is None:  # dropped only with every graph
            raise RuntimeError("ModernBERT CUDA graph replayed without its output buffer")
        tokens, _ = key
        slots = row_slots(tokens)
        rows = len(lengths)
        lengths_np = np.asarray(lengths, dtype=np.int32)
        ends = np.cumsum(lengths_np, dtype=np.int32)
        total = int(ends[-1])
        host = np.zeros(2 * tokens + slots + 1, dtype=np.int32)
        host[:total] = np.asarray(token_ids, dtype=np.int32)
        host[total:tokens] = self._pad_token_id
        # Positions restart at every row; padding positions stay 0.
        starts = ends - lengths_np
        host[tokens : tokens + total] = np.arange(total, dtype=np.int32) - np.repeat(starts, lengths_np)
        # cu_seqlens: the real rows, then empty slots at the real token count.
        cu_seqlens = host[2 * tokens :]
        cu_seqlens[1 : rows + 1] = ends
        cu_seqlens[rows + 1 :] = total
        # From pageable memory: the copy has read ``host`` when it returns.
        entry.inputs.copy_(torch.from_numpy(host), non_blocking=True)
        entry.graph.replay()
        return PackedForward(
            hidden=self._hidden_states(tokens)[:total],
            cu_seqlens=entry.inputs[2 * tokens : 2 * tokens + rows + 1],
            input_ids=entry.inputs[:total],
            lengths=list(lengths),
        )
