"""ModernBERT flash adapters replaying their encoder as CUDA graphs.

The CPU tests replace a recorded graph with a stand-in whose replay runs the
recorded encoder again on the graph's static inputs, and small references
stand in for flash-attn's kernels. So they check what a graph is fed and what
the adapters make of its output: the bucketed shapes, the packed layout
(padding tokens outside every sequence, empty slots, positions), the
recording policy, and the dense, late-interaction and reranking heads against
their eager forwards. The reference attention writes NaN to every row outside
a sequence, as the real kernel leaves them unwritten, so a padding row that
reached a real one would fail every comparison. The ``gpu_hw`` tests record
and replay real graphs.
"""

from __future__ import annotations

import logging
import sys
import types
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from sie_server.adapters import _modernbert_flash_graphs as graphs_module
from sie_server.adapters._cuda_graphs import RECORDING_LOCK
from sie_server.adapters._modernbert_flash_graphs import (
    PackedForward,
    VarlenGraphRunner,
    _Graph,
    bucketed_shapes,
    modernbert_encoder,
    parse_graph_mode,
    row_slots,
    seqlen_buckets,
    token_bound,
    token_buckets,
)
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter
from sie_server.adapters.gliclass import cuda_graphs as gliclass_graphs
from sie_server.adapters.modernbert_flash import ModernBERTFlashAdapter
from sie_server.adapters.modernbert_flash_cross_encoder import ModernBertFlashCrossEncoderAdapter
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options
from sie_server.types.inputs import Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.nn import functional
from transformers import ModernBertConfig, ModernBertForSequenceClassification, ModernBertModel, PreTrainedTokenizerFast

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_ADAPTER_PATHS = frozenset(
    {
        "sie_server.adapters.modernbert_flash:ModernBERTFlashAdapter",
        "sie_server.adapters.colbert_modernbert_flash.adapter:ColBERTModernBERTFlashAdapter",
        "sie_server.adapters.modernbert_flash_cross_encoder:ModernBertFlashCrossEncoderAdapter",
    }
)
# The shipped models that load with graphs: every one on these adapters, each
# of which passed the margin rule against eager forwards (see the server README).
_GRAPHS_BY_DEFAULT = {
    "Alibaba-NLP/gte-modernbert-base",
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "ibm-granite/granite-embedding-small-english-r2",
    "lightonai/GTE-ModernColBERT-v1",
    "lightonai/Reason-ModernColBERT",
    "lightonai/mLateOn",
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
    "nomic-ai/modernbert-embed-base",
    "topk-io/Iso-ModernColBERT",
}
_WINDOW = 64
_HIDDEN = 32
_WORDS = [
    "the",
    "a",
    "patient",
    "drug",
    "trial",
    "dose",
    "protein",
    "cell",
    "gene",
    "effect",
    "risk",
    "study",
    "model",
    "data",
    "result",
    "group",
]
# From one word to past the window, so rows pack against each other, span
# several sliding windows and get truncated.
_TEXTS = [" ".join(_WORDS[: 1 + i % len(_WORDS)] * (1 + i // 4)) for i in range(24)]


# -- reference kernels ------------------------------------------------------


def _reference_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """``flash_attn_varlen_func``: attention within each sequence; rows outside every sequence left unwritten (NaN)."""
    assert not causal
    assert torch.equal(cu_seqlens_q, cu_seqlens_k)
    bounds = cu_seqlens_q.tolist()
    lengths = [end - start for start, end in pairwise(bounds)]
    assert bounds[0] == 0
    assert all(length >= 0 for length in lengths)
    assert max(lengths) <= max_seqlen_q == max_seqlen_k
    out = torch.full_like(query, float("nan"))
    for start, end in pairwise(bounds):
        if start == end:
            continue
        rows = [t[start:end].transpose(0, 1) for t in (query, key, value)]
        mask = None
        if window_size != (-1, -1):
            positions = torch.arange(end - start, device=query.device)
            mask = (positions[:, None] - positions[None, :]).abs() <= window_size[0]
        attended = functional.scaled_dot_product_attention(*rows, attn_mask=mask, scale=softmax_scale)
        out[start:end] = attended.transpose(0, 1)
    return out


def _reference_qkvpacked(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """``flash_attn_varlen_qkvpacked_func`` on ``[tokens, 3, heads, dim]``."""
    return _reference_varlen(
        qkv[:, 0],
        qkv[:, 1],
        qkv[:, 2],
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=window_size,
    )


def _rotate(qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """flash-attn's non-interleaved rotation of ``[tokens, heads, dim]`` by per-token ``[tokens, dim / 2]`` tables."""
    half = qk.shape[-1] // 2
    x0, x1 = qk[..., :half].float(), qk[..., half:].float()
    cos, sin = cos[:, None, :].float(), sin[:, None, :].float()
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).to(qk.dtype)


def _reference_rotate_packed(qkv: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """``rotate_packed_qkv_`` in PyTorch."""
    positions = positions.long()
    qkv[:, :2] = torch.stack([_rotate(qkv[:, i], cos[positions], sin[positions]) for i in range(2)], dim=1)


class _FlashRotary(torch.nn.Module):
    """flash-attn's ``RotaryEmbedding`` as Hugging Face ModernBERT's flash-attention layers use it."""

    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _update_cos_sin_cache(self, seqlen: int, device: Any = None, dtype: Any = None) -> None:
        if seqlen > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seqlen
            freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), self.inv_freq)
            self._cos_cached, self._sin_cached = freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        positions = torch.cat([torch.arange(end - start) for start, end in pairwise(cu_seqlens.tolist())])
        _reference_rotate_packed(qkv, positions, self._cos_cached, self._sin_cached)
        return qkv


@pytest.fixture
def reference_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = _reference_varlen  # ty: ignore[unresolved-attribute]
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_varlen_qkvpacked_func = _reference_qkvpacked  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flash_attn", module)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)


# -- a runner whose graphs replay on the CPU ----------------------------------


class _Replay:
    """A recorded graph's stand-in: replaying runs the recorded encoder on the static inputs."""

    def __init__(self, runner: VarlenGraphRunner, inputs: torch.Tensor, key: tuple[int, int]) -> None:
        self._runner, self._inputs, self.key = runner, inputs, key
        self.replays = 0

    def replay(self) -> None:
        tokens, seqlen = self.key
        self._runner._hidden_states(tokens).copy_(self._runner._forward(self._inputs, tokens, seqlen))
        self.replays += 1

    def pool(self) -> None:
        return None


class _Runner(VarlenGraphRunner):
    """Records graphs as ``_Replay`` stand-ins. Time stands still unless a test moves ``now``.

    ``capture_error`` fails the next recording (every one with ``keep_failing``);
    the device has memory to spare unless a test clears ``headroom`` or sets a
    ``budget``.
    """

    def __init__(self, encode: Any, **kwargs: Any) -> None:
        self.now = 0.0
        kwargs.setdefault("hidden_size", _HIDDEN)
        kwargs.setdefault("dtype", torch.float32)
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("window", 8192)
        super().__init__(encode, clock=lambda: self.now, **kwargs)
        self.recorded: list[tuple[int, int]] = []
        self.capture_error: Exception | None = None
        self.keep_failing = False
        self.headroom = True
        self.budget = 2**40
        self.bytes_per_graph = 0

    def _has_headroom(self, device: torch.device) -> bool:
        return self.headroom

    def _memory_budget(self, device: torch.device) -> int:
        return self.budget

    def _capture(self, key: tuple[int, int]) -> _Graph:
        if self.capture_error is not None:
            error, self.capture_error = self.capture_error, (self.capture_error if self.keep_failing else None)
            raise error
        self.recorded.append(key)
        tokens, _ = key
        inputs = torch.zeros(2 * tokens + row_slots(tokens) + 1, dtype=torch.int32)
        self._hidden_states(tokens)
        return _Graph(graph=_Replay(self, inputs, key), inputs=inputs, device_bytes=self.bytes_per_graph)


def _echo_encode(
    ids: torch.Tensor, positions: torch.Tensor, cu: torch.Tensor, max_seqlen: int, total: int
) -> torch.Tensor:
    """An encoder whose output rows carry (token id, position) so layouts can be read back."""
    out = torch.zeros(total, _HIDDEN)
    out[:, 0] = ids.float()
    out[:, 1] = positions.float()
    return out


def _ids(lengths: list[int]) -> list[int]:
    return [100 + i for i in range(sum(lengths))]


def _layout(packed: PackedForward) -> tuple[list[int], list[int], list[int]]:
    return packed.hidden[:, 0].int().tolist(), packed.hidden[:, 1].int().tolist(), packed.cu_seqlens.tolist()


# -- shapes --------------------------------------------------------------------


class TestShapes:
    def test_token_buckets_are_powers_of_two_and_their_midpoints(self) -> None:
        assert token_buckets(1024) == (64, 128, 192, 256, 384, 512, 768, 1024)
        assert token_buckets(2048)[-3:] == (1024, 1536, 2048)
        assert token_buckets(512) == (64, 128, 192, 256, 384, 512)

    @pytest.mark.parametrize(("hidden_size", "tokens"), [(384, 2048), (768, 1024), (1024, 512)])
    def test_wider_encoders_hold_fewer_tokens(self, hidden_size: int, tokens: int) -> None:
        assert token_bound(hidden_size) == tokens

    @pytest.mark.parametrize(("tokens", "slots"), [(64, 8), (128, 8), (256, 16), (1024, 64), (2048, 128), (4096, 128)])
    def test_slots_grow_with_the_bucket(self, tokens: int, slots: int) -> None:
        assert row_slots(tokens) == slots

    @pytest.mark.parametrize(
        ("tokens", "window", "seqlens"),
        [
            (64, 8192, (64,)),
            (512, 8192, (512,)),
            (1024, 8192, (512, 1024)),
            (1024, 300, (300,)),
            (1024, 768, (512, 768)),
        ],
    )
    def test_short_and_long_row_graphs(self, tokens: int, window: int, seqlens: tuple[int, ...]) -> None:
        assert seqlen_buckets(tokens, window) == seqlens

    def test_the_shape_set_is_fixed_and_small(self) -> None:
        assert len(bucketed_shapes(1024, 8192)) == 10
        assert len(bucketed_shapes(2048, 8192)) == 14
        assert len(bucketed_shapes(512, 8192)) == 6

    @pytest.mark.parametrize(
        ("total", "rows", "longest", "key"),
        [
            (1, 1, 1, (64, 64)),
            (64, 4, 16, (64, 64)),
            (65, 1, 65, (128, 128)),
            (300, 1, 300, (384, 384)),
            (600, 2, 300, (768, 512)),
            (600, 1, 600, (768, 768)),
            (1024, 8, 128, (1024, 512)),
            # more rows than a bucket's slots move up a bucket
            (64, 9, 7, (192, 192)),
            (80, 20, 4, (384, 384)),
        ],
    )
    def test_a_forward_replays_the_smallest_graph_that_holds_it(
        self, total: int, rows: int, longest: int, key: tuple[int, int]
    ) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024, window=8192)
        assert runner.key(total, rows, longest) == key

    def test_past_the_bound_or_the_slots_runs_eagerly(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        assert runner.key(1025, 1, 64) is None
        assert runner.key(200, 65, 4) is None

    def test_graph_modes(self) -> None:
        assert parse_graph_mode("bucketed", adapter="A") == "bucketed"
        assert parse_graph_mode("off", adapter="A") == "off"
        assert parse_graph_mode(False, adapter="A") == "off"  # an unquoted YAML off
        for value in ("exact", "on", True, None, 1):
            with pytest.raises(ValueError, match="cuda_graphs must be 'off' or 'bucketed'"):
                parse_graph_mode(value, adapter="A")


# -- the packed layout ---------------------------------------------------------


class TestLayout:
    def test_padding_is_outside_every_sequence_and_positions_restart_per_row(self) -> None:
        seen: list[tuple[list[int], list[int], list[int], int]] = []

        def encode(
            ids: torch.Tensor, positions: torch.Tensor, cu: torch.Tensor, max_seqlen: int, total: int
        ) -> torch.Tensor:
            seen.append((ids.tolist(), positions.tolist(), cu.tolist(), max_seqlen))
            return _echo_encode(ids, positions, cu, max_seqlen, total)

        runner = _Runner(encode, max_tokens=1024, pad_token_id=7)
        lengths = [3, 1, 5]
        packed = runner.run(_ids(lengths), lengths, lambda packed: packed)
        assert packed is not None
        ids, positions, cu = seen[-1][:3]
        assert len(ids) == 64
        assert ids[:9] == _ids(lengths)
        assert ids[9:] == [7] * 55  # padding ids
        assert positions[:9] == [0, 1, 2, 0, 0, 1, 2, 3, 4]
        assert positions[9:] == [0] * 55
        # the real rows, then every other slot empty at the real token count
        assert cu == [0, 3, 4, 9] + [9] * (row_slots(64) - 3)
        assert seen[-1][3] == 64
        # the head sees the real rows only
        assert _layout(packed) == (_ids(lengths), [0, 1, 2, 0, 0, 1, 2, 3, 4], [0, 3, 4, 9])
        assert packed.input_ids.tolist() == _ids(lengths)
        assert packed.lengths == lengths

    def test_a_replay_rewrites_the_whole_layout(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        runner.run(_ids([40]), [40], lambda packed: None or 0)
        packed = runner.run([5, 6], [1, 1], lambda packed: packed)
        assert packed is not None
        assert _layout(packed) == ([5, 6], [0, 0], [0, 1, 2])
        assert runner.recorded == [(64, 64)]
        assert runner.stats.replayed == 1


# -- the recording policy -------------------------------------------------------


def _run(runner: _Runner, lengths: list[int]) -> Any:
    return runner.run(_ids(lengths), lengths, lambda packed: "graph")


def _lengths_for(key: tuple[int, int]) -> list[int]:
    """Rows that fill a graph of ``key`` exactly: one row of its ``max_seqlen``, then shorter ones."""
    tokens, seqlen = key
    lengths, left = [seqlen], tokens - seqlen
    while left:
        lengths.append(min(left, 512))
        left -= lengths[-1]
    return lengths


class TestPolicy:
    def test_a_shape_records_once_then_replays(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        assert _run(runner, [10]) == "graph"
        assert _run(runner, [20, 30]) == "graph"
        assert runner.recorded == [(64, 64)]
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 1)

    @pytest.mark.parametrize(
        ("lengths", "reason"),
        [([], "empty"), ([0], "empty"), ([_WINDOW + 1], "too_long"), ([60] * 20, "too_large")],
    )
    def test_forwards_a_graph_cannot_hold_run_eagerly(self, lengths: list[int], reason: str) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024, window=_WINDOW)
        assert _run(runner, lengths) is None
        assert runner.stats.eager == {reason: 1}
        assert runner.recorded == []

    def test_a_failed_shape_is_not_recorded_again_and_three_turn_graphs_off(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        runner.capture_error = RuntimeError("capture failed")
        assert _run(runner, [10]) is None
        assert _run(runner, [10]) is None
        assert runner.stats.eager == {"recording_failed": 1, "failed_shape": 1}
        assert _run(runner, [100]) == "graph"  # other shapes still record
        runner.capture_error, runner.keep_failing = RuntimeError("capture failed"), True
        with caplog.at_level(logging.ERROR):
            assert _run(runner, [200]) is None
            assert _run(runner, [300]) is None
        assert runner.disabled
        assert runner.graph_count == 0
        assert "off for this process" in caplog.text
        assert _run(runner, [100]) is None
        assert runner.stats.eager["disabled"] == 1
        assert runner.stats.recording_failures == 3

    def test_a_failed_first_replay_counts_as_a_failed_recording(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        original = runner._replay

        def failing(*args: Any) -> Any:
            raise RuntimeError("replay failed")

        runner._replay = failing  # ty: ignore[invalid-assignment]
        assert _run(runner, [10]) is None
        assert runner.graph_count == 0
        runner._replay = original  # ty: ignore[invalid-assignment]
        assert _run(runner, [10]) is None
        assert runner.stats.eager == {"recording_failed": 1, "failed_shape": 1}

    def test_head_errors_fail_the_forward_but_not_the_shape(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)

        def head(packed: PackedForward) -> Any:
            raise ValueError("head failed")

        for _ in range(4):
            with pytest.raises(ValueError, match="head failed"):
                runner.run(_ids([10]), [10], head)
        assert runner.stats.recording_failures == 0
        assert not runner.disabled
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 3)

    def test_running_out_of_memory_while_recording_drops_graphs_and_pauses(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        _run(runner, [10])
        runner.capture_error = torch.OutOfMemoryError("CUDA out of memory")
        assert _run(runner, [100]) is None
        assert runner.graph_count == 0
        assert runner.stats.drops == 1
        assert _run(runner, [10]) is None
        assert runner.stats.eager["recording_paused"] == 1
        runner.now += 62.0
        assert _run(runner, [10]) == "graph"
        assert 100 not in {key[0] for key in runner._failed}

    def test_out_of_memory_in_a_forward_drops_the_graphs(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        _run(runner, [10])

        def head(packed: PackedForward) -> Any:
            raise torch.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.OutOfMemoryError):
            runner.run(_ids([10]), [10], head)
        assert runner.graph_count == 0
        assert runner.stats.drops == 1
        assert _run(runner, [10]) == "graph"

    def test_recording_is_rationed(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=8192)
        shapes = sorted(bucketed_shapes(8192, 8192))
        for key in shapes[:16]:
            assert _run(runner, _lengths_for(key)) == "graph"
        assert _run(runner, _lengths_for(shapes[16])) is None
        assert runner.stats.eager == {"recording_paused": 1}
        runner.now += 2.0
        assert _run(runner, _lengths_for(shapes[16])) == "graph"
        assert runner.recorded == shapes[:17]

    def test_no_recording_without_headroom_or_while_another_model_records(self) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        runner.headroom = False
        assert _run(runner, [10]) is None
        runner.headroom = True
        assert RECORDING_LOCK.acquire(blocking=False)
        try:
            assert _run(runner, [10]) is None
        finally:
            RECORDING_LOCK.release()
        assert runner.stats.eager == {"no_headroom": 1, "busy": 1}
        assert _run(runner, [10]) == "graph"
        assert not RECORDING_LOCK.locked()

    def test_the_recording_lock_is_shared_with_the_gliclass_graphs(self) -> None:
        assert gliclass_graphs._RECORDING_LOCK is RECORDING_LOCK
        assert graphs_module.RECORDING_LOCK is RECORDING_LOCK

    def test_a_full_budget_stops_recording_and_keeps_the_graphs(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024)
        runner.bytes_per_graph = 10 * 2**20
        runner.budget = 15 * 2**20
        _run(runner, [10])
        with caplog.at_level(logging.WARNING):
            _run(runner, [100])  # over budget once kept
        assert "their memory budget" in caplog.text
        assert _run(runner, [300]) is None
        assert runner.stats.eager == {"budget_full": 1}
        assert _run(runner, [10]) == "graph"
        assert _run(runner, [100]) == "graph"
        assert runner.graph_count == 2

    def test_held_memory_counts_recordings_buffers_and_tables(self) -> None:
        table = torch.zeros(1024)
        runner = _Runner(_echo_encode, max_tokens=1024, static_tensors=[table])
        runner.bytes_per_graph = 1000
        _run(runner, [10])
        inputs = (2 * 64 + row_slots(64) + 1) * 4
        hidden = 1024 * _HIDDEN * 4
        assert runner.held_bytes == 1000 + inputs + hidden + table.nbytes
        runner.clear()
        assert runner.held_bytes == table.nbytes

    def test_counters_are_logged_every_ten_minutes(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner(_echo_encode, max_tokens=1024, name="m")
        _run(runner, [10])
        runner.now += 601.0
        with caplog.at_level(logging.INFO):
            _run(runner, [10])
        assert "ModernBERT CUDA graphs for m: 2 forwards" in caplog.text


# -- adapters ----------------------------------------------------------------


def _tokenizer() -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[Q]", "[D]"]
    vocab = {token: index for index, token in enumerate([*specials, *_WORDS, ".", ","])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106 -- a vocabulary entry, not a secret
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",  # noqa: S106
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
        model_max_length=_WINDOW,
    )


def _config(**extra: Any) -> ModernBertConfig:
    torch.manual_seed(0)
    return ModernBertConfig(
        vocab_size=32,
        hidden_size=_HIDDEN,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        global_attn_every_n_layers=3,
        local_attention=8,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        max_position_embeddings=_WINDOW,
        pad_token_id=0,
        attn_implementation="eager",
        **extra,
    )


def _sharpen(model: torch.nn.Module) -> None:
    """Larger attention weights, so a wrong position, window or RoPE base changes the outputs visibly."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "Wqkv" in name:
                parameter.mul_(8.0)


def _texts(count: int, offset: int = 0) -> list[Item]:
    return [Item(text=_TEXTS[(offset + i) % len(_TEXTS)]) for i in range(count)]


def _attach_cpu_graphs(adapter: Any, encode: Any, tables: list[torch.Tensor], window: int) -> _Runner:
    runner = _Runner(encode, max_tokens=1024, window=window, static_tensors=tables)
    adapter._graphs = runner
    return runner


def _dense_pair() -> tuple[ModernBERTFlashAdapter, ModernBERTFlashAdapter, _Runner]:
    model = ModernBertModel(_config()).eval()
    _sharpen(model)
    adapters = []
    for graphs in ("bucketed", "off"):
        adapter = ModernBERTFlashAdapter(
            "tiny", compute_precision="float32", max_seq_length=_WINDOW, cuda_graphs=graphs
        )
        adapter._model, adapter._tokenizer, adapter._device, adapter._dense_dim = model, _tokenizer(), "cpu", _HIDDEN
        adapters.append(adapter)
    encode, tables = modernbert_encoder(model, window=_WINDOW, dtype=torch.float32)
    runner = _attach_cpu_graphs(adapters[0], encode, tables, _WINDOW)
    return adapters[0], adapters[1], runner


def _colbert_pair() -> tuple[ColBERTModernBERTFlashAdapter, ColBERTModernBERTFlashAdapter, _Runner]:
    model = ModernBertModel(_config()).eval()
    _sharpen(model)
    chain = [torch.randn(16, _HIDDEN), torch.randn(8, 16)]
    adapters = []
    for graphs in ("bucketed", "off"):
        adapter = ColBERTModernBERTFlashAdapter(
            "tiny",
            token_dim=8,
            compute_precision="float32",
            max_seq_length=_WINDOW,
            query_max_length=12,
            doc_punctuation_skiplist=True,
            cuda_graphs=graphs,
        )
        adapter._model, adapter._tokenizer, adapter._device, adapter._dense_chain = model, _tokenizer(), "cpu", chain
        adapter._doc_skiplist_ids = {adapter._tokenizer.convert_tokens_to_ids(",")}
        adapters.append(adapter)
    encode, tables = modernbert_encoder(model, window=_WINDOW, dtype=torch.float32)
    runner = _attach_cpu_graphs(adapters[0], encode, tables, _WINDOW)
    return adapters[0], adapters[1], runner


def _reranker_pair(
    monkeypatch: pytest.MonkeyPatch, pooling: str
) -> tuple[ModernBertFlashCrossEncoderAdapter, ModernBertFlashCrossEncoderAdapter, _Runner]:
    from sie_server.adapters import modernbert_flash_cross_encoder as module

    monkeypatch.setattr(module, "rotate_packed_qkv_", _reference_rotate_packed)
    model = ModernBertForSequenceClassification(_config(num_labels=1, classifier_pooling=pooling)).eval()
    _sharpen(model)
    for layer in model.model.layers:
        base = 160000.0 if layer.attn.local_attention == (-1, -1) else 10000.0
        layer.attn.rotary_emb = _FlashRotary(_HIDDEN // 2, base)
    adapters = []
    for graphs in ("bucketed", "off"):
        adapter = ModernBertFlashCrossEncoderAdapter(
            "tiny", compute_precision="float32", max_seq_length=_WINDOW, cuda_graphs=graphs
        )
        adapter._model, adapter._tokenizer, adapter._device, adapter._dtype = model, _tokenizer(), "cpu", torch.float32
        adapter._num_heads, adapter._hidden_size, adapter._head_dim = 2, _HIDDEN, _HIDDEN // 2
        adapter._use_sigmoid, adapter._use_mean_pooling = True, pooling == "mean"
        adapters.append(adapter)
    encoded = adapters[0]._graph_encoder(_WINDOW)
    assert encoded is not None
    runner = _attach_cpu_graphs(adapters[0], *encoded, _WINDOW)
    return adapters[0], adapters[1], runner


@pytest.mark.usefixtures("reference_flash")
class TestAdapters:
    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    @pytest.mark.parametrize(("count", "offset"), [(1, 0), (1, 23), (5, 3), (12, 0)])
    def test_dense_vectors_match_eager(self, pooling: str, count: int, offset: int) -> None:
        graphed, eager, runner = _dense_pair()
        items = _texts(count, offset)
        options = {"pooling": pooling}
        for _ in range(2):  # recorded, then replayed
            got = graphed.encode(items, ["dense"], options=options)
            want = eager.encode(items, ["dense"], options=options)
            np.testing.assert_allclose(got.dense, want.dense, atol=1e-5)
            assert got.extra["input_token_counts"] == want.extra["input_token_counts"]
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 1)

    @pytest.mark.parametrize("is_query", [True, False])
    @pytest.mark.parametrize(("count", "offset"), [(1, 5), (4, 0), (9, 11)])
    def test_token_vectors_match_eager(self, is_query: bool, count: int, offset: int) -> None:
        graphed, eager, runner = _colbert_pair()
        items = _texts(count, offset)
        for _ in range(2):
            got = graphed.encode(items, ["multivector"], is_query=is_query)
            want = eager.encode(items, ["multivector"], is_query=is_query)
            assert [g.shape for g in got.multivector] == [w.shape for w in want.multivector]
            for g, w in zip(got.multivector, want.multivector, strict=True):
                np.testing.assert_allclose(g, w, atol=1e-5)
            assert got.extra == want.extra
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 1)

    def test_maxsim_scores_match_eager(self) -> None:
        graphed, eager, _ = _colbert_pair()
        query, docs = Item(text=_TEXTS[3]), _texts(6, 7)
        np.testing.assert_allclose(graphed.score(query, docs), eager.score(query, docs), atol=1e-4)

    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    @pytest.mark.parametrize(("count", "offset"), [(1, 0), (3, 9), (8, 2)])
    def test_rerank_scores_match_eager(
        self, monkeypatch: pytest.MonkeyPatch, pooling: str, count: int, offset: int
    ) -> None:
        graphed, eager, runner = _reranker_pair(monkeypatch, pooling)
        queries, docs = _texts(count, offset), _texts(count, offset + 5)
        for _ in range(2):
            got = graphed.score_pairs(queries, docs)
            want = eager.score_pairs(queries, docs)
            np.testing.assert_allclose(got.scores, want.scores, atol=1e-5)
        np.testing.assert_allclose(graphed.score(queries[0], docs), eager.score(queries[0], docs), atol=1e-5)
        assert runner.stats.forwards == runner.stats.recorded + runner.stats.replayed == 3

    def test_rows_past_the_bound_run_eagerly(self) -> None:
        graphed, eager, runner = _dense_pair()
        runner._max_tokens, runner._tokens = 64, token_buckets(64)
        items = _texts(12)
        np.testing.assert_allclose(
            graphed.encode(items, ["dense"]).dense, eager.encode(items, ["dense"]).dense, atol=1e-5
        )
        assert runner.stats.eager == {"too_large": 1}

    def test_a_model_with_lora_adapters_loaded_runs_eagerly(self) -> None:
        graphed, _, runner = _dense_pair()
        graphed._peft_model = SimpleNamespace()  # ty: ignore[invalid-assignment]
        graphed.encode(_texts(2), ["dense"])
        assert runner.stats.forwards == 0

    def test_unload_drops_the_graphs(self) -> None:
        graphed, _, _ = _dense_pair()
        graphed.unload()
        assert graphed._graphs is None


class TestOptions:
    @pytest.mark.parametrize(
        "adapter", [ModernBERTFlashAdapter, ColBERTModernBERTFlashAdapter, ModernBertFlashCrossEncoderAdapter]
    )
    def test_cuda_graphs_is_a_load_time_option(self, adapter: Any) -> None:
        reject_unknown_loadtime_options(adapter, {"cuda_graphs": "bucketed"}, model_name="m")
        assert adapter("m", cuda_graphs="bucketed")._cuda_graphs == "bucketed"
        assert adapter("m", cuda_graphs=False)._cuda_graphs == "off"
        assert adapter("m")._cuda_graphs == "off"
        with pytest.raises(ValueError, match="cuda_graphs must be 'off' or 'bucketed'"):
            adapter("m", cuda_graphs="exact")

    def test_off_cuda_the_adapters_build_no_runner(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = ModernBERTFlashAdapter("m", cuda_graphs="bucketed")
        adapter._model = ModernBertModel(_config()).eval()
        adapter._tokenizer, adapter._device = _tokenizer(), "cpu"
        with caplog.at_level(logging.WARNING):
            assert adapter._graph_runner(torch.float32) is None
        assert "CUDA graphs need a CUDA device" in caplog.text

    def test_profiles_on_other_workers_do_not_inherit_graphs(self) -> None:
        """A profile that extends a graphed default onto the Candle worker must set its own load-time options.

        The Candle worker refuses load-time options it does not know, and a
        profile without its own load-time options inherits the default's.
        """
        for config in load_model_configs(_MODELS_DIR).values():
            for name in config.profiles:
                profile = config.resolve_profile(name)
                if not profile.adapter_path.startswith("sie_server."):
                    assert "cuda_graphs" not in profile.loadtime, f"{config.sie_id} profile {name}"

    def test_shipped_profiles_load_with_graphs(self) -> None:
        """Every model on these adapters ships with graphs: each passed the margin rule (see the server README)."""
        graphed = set()
        for config in load_model_configs(_MODELS_DIR).values():
            for name in config.profiles:
                profile = config.resolve_profile(name)
                if profile.adapter_path in _ADAPTER_PATHS:
                    assert parse_graph_mode(profile.loadtime.get("cuda_graphs"), adapter=profile.adapter_path) == (
                        "bucketed"
                    ), f"{config.sie_id} profile {name}"
                    graphed.add(config.sie_id.split(":")[0])
        assert graphed == _GRAPHS_BY_DEFAULT


# -- real graphs -------------------------------------------------------------


def _cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    from sie_server.core.inference import is_flash_attention_available

    return is_flash_attention_available("cuda:0")


@pytest.mark.gpu_hw
class TestOnGpu:
    def test_recorded_graphs_match_eager(self) -> None:
        if not _cuda_ready():
            pytest.skip("requires CUDA and flash-attn")
        model = ModernBertModel(_config()).eval().to("cuda", torch.bfloat16)
        adapters = []
        for graphs in ("bucketed", "off"):
            adapter = ModernBERTFlashAdapter("tiny", max_seq_length=_WINDOW, cuda_graphs=graphs)
            adapter._model, adapter._tokenizer, adapter._device, adapter._dense_dim = (
                model,
                _tokenizer(),
                "cuda:0",
                _HIDDEN,
            )
            adapter._graphs = adapter._graph_runner(torch.bfloat16)
            adapters.append(adapter)
        graphed, eager = adapters
        assert graphed._graphs is not None
        for count, offset in ((1, 0), (5, 3), (12, 0), (1, 23)):
            items = _texts(count, offset)
            for _ in range(2):
                got = graphed.encode(items, ["dense"]).dense
                want = eager.encode(items, ["dense"]).dense
                cosine = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
                assert cosine.min() > 0.999
        stats = graphed._graphs.stats
        assert stats.recorded >= 1
        assert stats.replayed >= 4
        assert stats.recording_failures == 0
        assert graphed._graphs.held_bytes > 0
