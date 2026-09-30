"""TopK-Embed CUDA graphs: shapes, recording policy, numerics of the recorded forward, adapter wiring.

These run without a GPU: recording and replay are replaced by a fake graph whose
replay runs the recorded forward eagerly, so the runner's own code decides what to
record and replay, and the padded forward it would record is checked against the
packed one. Real graphs are checked on a GPU by the Modal harness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sie_server.adapters._cuda_graphs import RECORDING_LOCK
from sie_server.adapters.topk_embed import graphs, packed
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter
from sie_server.types.inputs import Item

from .test_topk_embed import HIDDEN, PAD_ID, _fake_vision, _png, make_adapter
from .test_topk_embed_packed import _IdentityText, tiny_qwen3_5_text_model


class _FakeGraph:
    """Stands in for ``torch.cuda.CUDAGraph``: a replay runs the recorded computation eagerly."""

    def __init__(self, replay: Callable[[], Any]) -> None:
        self._replay = replay
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self._replay()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Runner(graphs.GraphRunner):
    """Records into fake graphs on the CPU; everything else is the runner's own code."""

    def __init__(
        self,
        text_model: Any,
        embed: Any,
        *,
        free: int = 10**12,
        total: int = 10**12,
        device_bytes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_model, embed, **kwargs)
        self.free, self.total, self.device_bytes = free, total, device_bytes
        self.fail_with: BaseException | None = None
        self.captured: list[graphs.Key] = []

    def _free_memory(self) -> int:
        return self.free

    def _total_memory(self) -> int:
        return self.total

    def _capture(self, key: graphs.Key) -> Any:
        self.captured.append(key)
        if self.fail_with is not None:
            raise self.fail_with
        rows, length = key
        input_ids = torch.full((rows, length), self._pad_token_id, dtype=torch.long)
        mask = torch.ones((rows, length), dtype=torch.bool)
        hidden = self._hidden_states(key)
        graph = _FakeGraph(lambda: hidden.copy_(self._forward(input_ids, mask)))
        return graphs._Graph(graph=graph, input_ids=input_ids, mask=mask, device_bytes=self.device_bytes)


def _identity(embeds: torch.Tensor, positions: torch.Tensor, layout: Any) -> torch.Tensor:
    return embeds


def _runner(max_tokens: int = 256, clock: Callable[[], float] | None = None, **kwargs: Any) -> _Runner:
    torch.manual_seed(0)
    embed = torch.nn.Embedding(64, HIDDEN)
    return _Runner(
        _identity, embed, pad_token_id=PAD_ID, max_tokens=max_tokens, clock=clock or _Clock(), name="test", **kwargs
    )


def _rows(*lengths: int) -> list[torch.Tensor]:
    # Token ids within the stand-in embedding's 64, never the pad id.
    return [torch.arange(n) % 63 + 1 for n in lengths]


class TestShapes:
    @pytest.mark.parametrize(
        ("length", "bucketed"),
        [(1, 16), (16, 16), (17, 32), (128, 128), (129, 192), (500, 512), (513, 768), (1000, 1024)],
    )
    def test_lengths_round_up_to_their_bucket(self, length: int, bucketed: int) -> None:
        assert graphs.bucket_length(length) == bucketed

    def test_row_buckets_are_powers_of_two_and_the_most_that_fit(self) -> None:
        assert graphs.row_buckets(16, 1536) == (1, 2, 4, 8, 16, 32, 64, 96)
        assert graphs.row_buckets(512, 1536) == (1, 2, 3)
        assert graphs.row_buckets(2048, 1536) == ()

    def test_keys(self) -> None:
        assert graphs.graph_key(3, 20, 1536) == (4, 32)
        assert graphs.graph_key(1, 1, 1536) == (1, 16)
        assert graphs.graph_key(96, 16, 1536) == (96, 16)
        assert graphs.graph_key(97, 16, 1536) is None  # too many rows at that length
        assert graphs.graph_key(1, 1600, 1536) is None  # too long for any graph

    @pytest.mark.parametrize(("hidden", "tokens"), [(512, 2048), (768, 2048), (1024, 1536), (2048, 768)])
    def test_wider_models_hold_fewer_tokens(self, hidden: int, tokens: int) -> None:
        assert graphs.default_max_tokens(hidden) == tokens

    @pytest.mark.parametrize("max_tokens", [768, 1536])
    def test_the_shapes_are_a_small_fixed_set_within_the_bound(self, max_tokens: int) -> None:
        shapes = graphs.bucketed_shapes(max_tokens)
        assert len(shapes) < 100
        assert all(rows * length <= max_tokens for rows, length in shapes)
        for rows in range(1, 20):
            for length in range(1, 200, 7):
                key = graphs.graph_key(rows, length, max_tokens)
                assert key is None or key in shapes


class TestRecordingPolicy:
    def test_a_shape_records_once_and_then_replays(self) -> None:
        runner = _runner()
        assert runner.run(_rows(20, 5, 7)) is not None  # three rows pad to four
        assert runner.run(_rows(25, 3, 9, 1)) is not None  # also (4, 32): same graph
        assert runner.captured == [(4, 32)]
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 1)

    def test_each_row_gets_its_own_vectors_right_padded(self) -> None:
        runner = _runner()
        rows = _rows(3, 9)
        hidden = runner.run(rows)
        assert hidden is not None
        assert hidden.shape == (2, 16, HIDDEN)
        embed = cast("torch.nn.Embedding", runner._embed)
        for i, row in enumerate(rows):
            torch.testing.assert_close(hidden[i, : len(row)], embed(row))
            torch.testing.assert_close(hidden[i, len(row) :], embed(torch.full((16 - len(row),), PAD_ID)))

    def test_padding_rows_attend_to_their_first_position(self) -> None:
        runner = _runner()
        runner.run(_rows(3, 5, 2))  # three rows in a graph of four
        entry = runner._graphs[(4, 16)]
        assert entry.mask[:3].sum(dim=1).tolist() == [3, 5, 2]
        assert entry.mask[3].tolist() == [True] + [False] * 15

    def test_past_the_token_bound_runs_eagerly(self) -> None:
        runner = _runner(max_tokens=64)
        assert runner.run(_rows(65)) is None
        assert runner.run(_rows(*[4] * 5)) is None  # five rows of 16 tokens is 80
        assert runner.stats.eager == {"too_large": 2}
        assert runner.captured == []

    def test_recording_is_rationed_in_time(self) -> None:
        clock = _Clock()
        runner = _runner(max_tokens=4096, clock=clock)
        lengths = [16 * step for step in range(1, 9)] + [128 + 64 * step for step in range(1, 7)] + [768, 1024]
        for length in lengths:  # 16 shapes, the burst
            assert runner.run(_rows(length)) is not None
        assert runner.run(_rows(1280)) is None
        assert runner.stats.eager == {"recording_paused": 1}
        clock.now += 2.0
        assert runner.run(_rows(1280)) is not None

    def test_one_recording_at_a_time_in_the_process(self) -> None:
        runner = _runner()
        assert RECORDING_LOCK.acquire(blocking=False)  # another model is recording
        try:
            assert runner.run(_rows(5)) is None
        finally:
            RECORDING_LOCK.release()
        assert runner.stats.eager == {"busy": 1}
        assert runner.run(_rows(5)) is not None
        assert not RECORDING_LOCK.locked()

    def test_no_recording_while_device_memory_is_short(self) -> None:
        runner = _runner(free=9 * 10**10, total=10**12)  # 9% free
        assert runner.run(_rows(5)) is None
        assert runner.stats.eager == {"no_headroom": 1}

    def test_over_the_budget_it_stops_recording_and_keeps_its_graphs(self) -> None:
        runner = _runner(total=10**9, free=10**9, device_bytes=5 * 10**7)  # 5% of the device per graph
        assert runner.run(_rows(5)) is not None
        assert runner.run(_rows(40)) is None
        assert runner.stats.eager == {"budget_full": 1}
        assert runner.run(_rows(7)) is not None  # the recorded shape still replays
        assert runner.stats.replayed == 1

    def test_a_shape_that_fails_to_record_runs_eagerly_from_then_on(self) -> None:
        runner = _runner()
        runner.fail_with = RuntimeError("capture failed")
        assert runner.run(_rows(5)) is None
        assert runner.run(_rows(6)) is None
        assert runner.stats.eager == {"recording_failed": 1, "failed_shape": 1}
        assert runner.stats.recording_failures == 1
        runner.fail_with = None
        assert runner.run(_rows(20)) is not None  # other shapes still record

    def test_three_shapes_that_fail_turn_graphs_off(self) -> None:
        runner = _runner()
        assert runner.run(_rows(5)) is not None
        runner.fail_with = RuntimeError("capture failed")
        for length in (20, 40, 60):
            assert runner.run(_rows(length)) is None
        assert runner.disabled
        assert runner.graph_count == 0
        assert runner.run(_rows(5)) is None
        assert runner.stats.eager["disabled"] == 1

    def test_running_out_of_memory_while_recording_drops_the_graphs_and_pauses(self) -> None:
        clock = _Clock()
        runner = _runner(clock=clock)
        assert runner.run(_rows(5)) is not None
        runner.fail_with = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 MiB")
        assert runner.run(_rows(40)) is None
        assert runner.graph_count == 0
        assert runner.stats.drops == 1
        assert runner.stats.recording_failures == 0  # memory is not the shape's fault
        runner.fail_with = None
        assert runner.run(_rows(5)) is None
        assert runner.stats.eager["recording_paused"] == 1
        clock.now += 60.0
        assert runner.run(_rows(5)) is not None

    def test_every_forward_counts_once(self) -> None:
        runner = _runner(max_tokens=64)
        for lengths in ((5,), (5,), (70,), (6, 7)):
            runner.run(_rows(*lengths))
        assert runner.stats.forwards == 4


class TestWarmUp:
    def test_runs_the_padded_forward_at_each_step_up_to_the_bound(self) -> None:
        runner = _runner(max_tokens=2500)
        shapes: list[tuple[int, ...]] = []

        def forward(input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            assert bool(mask.all())
            shapes.append(tuple(input_ids.shape))
            return torch.zeros(*input_ids.shape, HIDDEN)

        with patch.object(runner, "_forward", side_effect=forward):
            runner.warm_up(1024)
        assert shapes == [(1, 1024), (1, 2048), (1, 2500)]
        assert runner.stats.forwards == 0  # nothing recorded, nothing counted


class TestRecordedForward:
    """The padded forward a graph records gives each input what the packed forward gives it."""

    def test_matches_the_packed_forward(self) -> None:
        model = tiny_qwen3_5_text_model()
        text = packed.PackedTextModel(model, packed.reference_kernels())
        runner = _Runner(text, model.embed_tokens, pad_token_id=0, max_tokens=256, clock=_Clock())
        torch.manual_seed(9)
        rows = [torch.randint(1, 32, (n,)) for n in (5, 70, 1)]  # 70 spans two delta-rule chunks
        with torch.inference_mode():
            hidden = runner.run(rows)
            assert hidden is not None
            lengths = [len(row) for row in rows]
            positions = torch.cat([torch.arange(n) for n in lengths]).view(1, 1, -1).expand(3, 1, -1)
            ours = text(
                model.embed_tokens(torch.cat(rows).unsqueeze(0)), positions, packed.Packing.from_lengths(lengths, "cpu")
            )
        start = 0
        for i, n in enumerate(lengths):
            torch.testing.assert_close(hidden[i, :n], ours[0, start : start + n], rtol=1e-4, atol=1e-4)
            start += n


class _Graphs:
    """Stands in for the adapter's runner: serves right-padded embeddings, or declines."""

    def __init__(self, embed: torch.nn.Embedding, *, serve: bool = True, error: BaseException | None = None) -> None:
        self.embed, self.serve, self.error = embed, serve, error
        self.calls: list[list[int]] = []
        self.cleared = 0

    def run(self, rows: list[torch.Tensor]) -> torch.Tensor | None:
        self.calls.append([len(row) for row in rows])
        if self.error is not None:
            raise self.error
        if not self.serve:
            return None
        length = graphs.bucket_length(max(len(row) for row in rows))
        ids = torch.full((len(rows), length), PAD_ID)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
        return self.embed(ids)

    def clear(self) -> None:
        self.cleared += 1


class TestAdapterWiring:
    def test_the_mode_is_checked_at_construction(self) -> None:
        assert make_adapter(cuda_graphs=False)._cuda_graphs == "off"
        assert make_adapter(cuda_graphs="bucketed")._cuda_graphs == "bucketed"
        with pytest.raises(ValueError, match="cuda_graphs"):
            make_adapter(cuda_graphs="sometimes")

    def test_graphs_stay_off_without_cuda_or_the_fast_kernels(self) -> None:
        adapter = make_adapter(cuda_graphs="bucketed")
        assert adapter._graph_runner("cpu") is None
        adapter._packed_text = cast("Any", _IdentityText())
        with patch.object(adapter._packed_text, "kernels", packed.reference_kernels(), create=True):
            assert adapter._graph_runner("cuda:0") is None  # reference kernels only

    @pytest.fixture
    def adapters(self) -> tuple[TopkEmbedAdapter, TopkEmbedAdapter]:
        graphed = make_adapter(packed=True)
        graphed._packed_text = cast("Any", _IdentityText())
        graphed._graphs = cast("Any", _Graphs(graphed._model.get_input_embeddings()))
        return make_adapter(), graphed

    def test_text_uses_the_graphs(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        padded, graphed = adapters
        items = [Item(text="a short one"), Item(text="another, slightly longer, document.")]
        ours = graphed.encode(items, ["multivector"])
        reference = padded.encode(items, ["multivector"])
        assert ours.multivector is not None
        assert reference.multivector is not None
        for a, b in zip(ours.multivector, reference.multivector, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        runner = cast("_Graphs", graphed._graphs)
        assert len(runner.calls) == 1
        assert cast("_IdentityText", graphed._packed_text).calls == []  # the eager packed path never ran

    def test_a_runner_that_declines_leaves_the_packed_path(
        self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]
    ) -> None:
        _, graphed = adapters
        cast("_Graphs", graphed._graphs).serve = False
        graphed.encode([Item(text="a short one")], ["multivector"])
        assert len(cast("_IdentityText", graphed._packed_text).calls) == 1

    def test_images_skip_the_graphs(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        _, graphed = adapters
        with _fake_vision(graphed):
            graphed.encode([Item(images=[_png()])], ["multivector"])
        assert cast("_Graphs", graphed._graphs).calls == []

    def test_running_out_of_memory_drops_the_graphs(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        _, graphed = adapters
        runner = cast("_Graphs", graphed._graphs)
        runner.error = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 MiB")
        with pytest.raises(torch.cuda.OutOfMemoryError):
            graphed.encode([Item(text="a short one")], ["multivector"])
        assert runner.cleared == 1

    def test_other_errors_keep_the_graphs(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        _, graphed = adapters
        runner = cast("_Graphs", graphed._graphs)
        runner.error = RuntimeError("something else")
        with pytest.raises(RuntimeError, match="something else"):
            graphed.encode([Item(text="a short one")], ["multivector"])
        assert runner.cleared == 0

    def test_unload_drops_the_graphs(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        _, graphed = adapters
        runner = cast("_Graphs", graphed._graphs)
        graphed.unload()
        assert graphed._graphs is None
        assert runner.cleared == 1
