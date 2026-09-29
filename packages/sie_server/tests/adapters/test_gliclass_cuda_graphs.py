"""GLiClass CUDA graph policy: which forwards replay a graph, which record one, which run eagerly.

These run without a GPU: recording and replay are replaced by fakes. Scores
from real graphs are checked against eager scores in test_gliclass_parity.py
(``gpu_hw``).
"""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from gliclass.model import GLiClassUniEncoder
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.adapters.gliclass import cuda_graphs as cuda_graphs_module
from sie_server.adapters.gliclass.cuda_graphs import (
    CudaGraphRunner,
    batch_buckets,
    bucket_width,
    bucketed_shapes,
    segment_ids,
    token_bound,
    unsupported_reason,
)
from sie_server.core.loader import _build_adapter_kwargs, load_model_configs, reject_unknown_loadtime_options
from sie_server.types.inputs import InvalidInputError

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# The shipped profiles that load with graphs: DeBERTa-v3 models whose bucketed
# scores moved no probability by more than 0.02 against eager execution and
# changed a top label only where eager's top two were closer than eager's own
# batching noise (see "GLiClass CUDA graphs" in the server README).
_BUCKETED_BY_DEFAULT = {
    "knowledgator/gliclass-small-v1.0",
    "knowledgator/gliclass-base-v1.0",
    "knowledgator/gliclass-large-v1.0",
    "knowledgator/gliclass-base-v3.0",
    "knowledgator/gliclass-large-v3.0",
    "knowledgator/gliclass-instruct-base-v1.0",
    "knowledgator/gliclass-instruct-large-v1.0",
    "knowledgator/opir-multitask-large-v1.0",
}


class _Encoder:
    def __init__(self) -> None:
        self.built: list[int] = []

    def get_rel_pos(self, hidden_states: torch.Tensor, query_states: Any = None, relative_pos: Any = None) -> Any:
        self.built.append(hidden_states.shape[-2])
        return ("built", hidden_states.shape[-2])


def _model(
    *,
    encoder_type: str = "deberta-v2",
    architecture: str = "uni-encoder",
    pooling: str = "first",
    segment_embeddings: bool = False,
    hidden_size: int = 1024,
    head: bool = True,
) -> Any:
    config = SimpleNamespace(
        architecture_type=architecture,
        encoder_config=SimpleNamespace(model_type=encoder_type, hidden_size=hidden_size),
        pooling_strategy=pooling,
        use_lstm=False,
        use_segment_embeddings=segment_embeddings,
        example_token_index=9,
        text_token_index=8,
    )
    inner = SimpleNamespace(encoder_model=SimpleNamespace(encoder=_Encoder()), _create_segment_ids=lambda ids: "eager")
    if head:
        inner.process_encoder_output = lambda ids, mask, layer, labels=None, max_num_classes=None: ("head", layer)
    return SimpleNamespace(config=config, model=inner)


class _Runner(CudaGraphRunner):
    """Records and replays graphs with fakes; the runner's own code scores them.

    A graph replay (``_replay_graph``) returns hidden states for the forward's
    real rows, or raises ``replay_error``. The scoring head answers with ones,
    ``(batch, classes)``: the forward's real rows and label slots, or raises
    ``head_error``. A recording whose capture fails (``capture_error``) raises
    that error. Time stands still unless a test moves ``now``; the device
    always has memory to spare unless a test clears ``headroom`` or sets a
    ``budget``.
    """

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        self.now = 0.0
        kwargs.setdefault("max_length", 512)
        super().__init__(model or _model(), pad_token_id=0, clock=lambda: self.now, **kwargs)
        self.recorded: list[tuple[int, int]] = []
        self.replayed: list[tuple[int, int]] = []
        self.head_classes: list[int | None] = []
        self.capture_error: Exception | None = None
        self.headroom = True
        self.budget = 2**40
        self.bytes_per_graph = 0
        self.replay_error: Exception | None = None
        self.head_error: Exception | None = None
        self.buffers_alive = False
        self._head = self._fake_head

    def _has_headroom(self, device: torch.device) -> bool:
        return self.headroom

    def _memory_budget(self, device: torch.device) -> int:
        return self.budget

    # Whether ``capture_error`` fails every recording, not just the next one.
    keep_failing = False

    def _record(self, key: Any, inputs: dict[str, torch.Tensor]) -> Any:
        if self.capture_error is not None:
            error, self.capture_error = self.capture_error, (self.capture_error if self.keep_failing else None)
            raise error
        self.recorded.append(key)
        self._relative_pos.setdefault(key[1], torch.zeros(1))
        self.buffers_alive = True  # like the hidden-state buffer a real recording writes
        # Static inputs with a row per padded row and no memory to count.
        static = {name: torch.zeros(key[0], 0, dtype=torch.long) for name in ("input_ids", "attention_mask")}
        return SimpleNamespace(key=key, device_bytes=self.bytes_per_graph, inputs=static, graph=None)

    def clear(self) -> None:
        super().clear()
        self.buffers_alive = False

    def _replay_graph(self, key: Any, entry: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.buffers_alive, "replayed a graph after its buffers were dropped"
        if self.replay_error is not None:
            raise self.replay_error
        self.replayed.append(entry.key)
        return torch.zeros(inputs["input_ids"].shape[0], key[1], 1)

    def _fake_head(
        self, input_ids: Any, attention_mask: Any, hidden: torch.Tensor, labels: Any, max_num_classes: int | None
    ) -> tuple[torch.Tensor]:
        assert input_ids.shape[0] == attention_mask.shape[0] == hidden.shape[0]
        if self.head_error is not None:
            raise self.head_error
        self.head_classes.append(max_num_classes)
        return (torch.ones(hidden.shape[0], max_num_classes or 1),)


def _inputs(batch: int, length: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.ones(batch, length, dtype=torch.long),
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
    }


class TestSupport:
    def test_deberta_uni_encoders_on_cuda_are_supported(self) -> None:
        assert unsupported_reason(_model(), "cuda:0") is None

    @pytest.mark.parametrize(
        ("model", "device", "reason"),
        [
            (_model(), "cpu", "CUDA device"),
            (_model(), "mps", "CUDA device"),
            (_model(encoder_type="modernbert"), "cuda:0", "modernbert encoder"),
            (_model(architecture="bi-encoder"), "cuda:0", "uni-encoder"),
            (_model(head=False), "cuda:0", "scoring head"),
        ],
    )
    def test_other_models_and_devices_run_eagerly(self, model: Any, device: str, reason: str) -> None:
        message = unsupported_reason(model, device)
        assert message is not None
        assert reason in message


class TestShapes:
    @pytest.mark.parametrize(("max_length", "width"), [(512, 32), (1024, 64), (128, 16)])
    def test_bucket_width_follows_the_window(self, max_length: int, width: int) -> None:
        assert bucket_width(max_length) == width

    @pytest.mark.parametrize(("hidden_size", "tokens"), [(384, 2048), (768, 2048), (1024, 1024), (1536, 1024)])
    def test_wider_encoders_hold_fewer_tokens(self, hidden_size: int, tokens: int) -> None:
        assert token_bound(hidden_size) == tokens

    @pytest.mark.parametrize(("length", "bucketed"), [(1, 32), (32, 32), (33, 64), (500, 512), (512, 512)])
    def test_bucketed_lengths_round_up_to_the_bucket(self, length: int, bucketed: int) -> None:
        runner = _Runner()

        assert runner.key(1, length, "bucketed") == (1, bucketed)
        assert runner.key(1, length, "exact") == (1, length)

    @pytest.mark.parametrize(
        ("length", "tokens", "sizes"),
        [
            (32, 2048, (1, 2, 4, 8, 16, 32, 64)),
            (96, 2048, (1, 2, 4, 8, 16, 21)),
            (512, 2048, (1, 2, 4)),
            (96, 1024, (1, 2, 4, 8, 10)),
            (192, 1024, (1, 2, 4, 5)),
            (512, 1024, (1, 2)),
            (1024, 1024, (1,)),
        ],
    )
    def test_batch_buckets_are_powers_of_two_and_the_largest_that_fits(
        self, length: int, tokens: int, sizes: tuple[int, ...]
    ) -> None:
        assert batch_buckets(length, tokens) == sizes

    def test_bucketed_batches_pad_up_to_their_bucket(self) -> None:
        runner = _Runner()  # a DeBERTa-large-wide encoder: 1,024 tokens

        assert [runner.key(batch, 100, "bucketed") for batch in (1, 2, 3, 4, 5, 8)] == [
            (1, 128),
            (2, 128),
            (4, 128),
            (4, 128),
            (8, 128),
            (8, 128),
        ]
        assert runner.key(9, 100, "bucketed") is None  # 9 x 128 tokens: over the bound
        assert runner.key(5, 190, "bucketed") == (5, 192)  # the largest batch at 192 tokens
        assert runner.key(3, 100, "exact") == (3, 100)  # exact mode never pads

    def test_shapes_past_the_token_bound_run_eagerly(self) -> None:
        runner = _Runner()

        assert runner.key(2, 512, "exact") == (2, 512)
        assert runner.key(3, 512, "exact") is None
        assert runner.key(32, 32, "bucketed") == (32, 32)
        assert runner.key(33, 32, "bucketed") is None
        assert runner.key(5, 200, "bucketed") is None  # 5 x 224 tokens

        narrow = _Runner(_model(hidden_size=768))
        assert narrow.key(4, 512, "exact") == (4, 512)
        assert narrow.key(9, 100, "bucketed") == (16, 128)
        assert narrow.key(65, 32, "bucketed") is None

    def test_the_token_bound_does_not_grow_with_the_window(self) -> None:
        runner = CudaGraphRunner(_model(), pad_token_id=0, max_length=1024)

        assert runner.key(1, 1024, "exact") == (1, 1024)
        assert runner.key(2, 1024, "exact") is None
        assert runner.key(1, 600, "bucketed") == (1, 640)  # 64-token buckets

    @pytest.mark.parametrize(
        ("max_length", "hidden_size", "count"),
        [(1024, 1024, 33), (1024, 768, 52), (512, 1024, 52), (512, 768, 71)],
        ids=["large-1024", "base-1024", "large-512", "base-512"],
    )
    def test_every_bucketed_forward_maps_into_a_small_fixed_set(
        self, max_length: int, hidden_size: int, count: int
    ) -> None:
        runner = _Runner(_model(hidden_size=hidden_size), max_length=max_length)
        shapes = bucketed_shapes(max_length, token_bound(hidden_size))

        assert len(shapes) == count
        assert runner.shape_count == count
        keys = {runner.key(batch, length, "bucketed") for batch in range(1, 129) for length in range(1, max_length + 1)}
        assert keys - {None} == shapes

    @pytest.mark.parametrize("pooling", ["avg", "last", "max"])
    def test_poolings_that_read_padding_keep_exact_shapes(self, pooling: str) -> None:
        runner = _Runner(_model(pooling=pooling))

        assert runner.key(3, 100, "bucketed") == (3, 100)
        assert runner.shape_count is None  # unbounded: a most-recently-used cache


class TestRecordingPolicy:
    def test_off_never_records(self) -> None:
        runner = _Runner()

        assert runner.run(_inputs(1, 100), 4, "off") is None
        assert runner.recorded == []
        assert runner.stats.forwards == 0  # opted-out forwards are not offered to the runner

    def test_exact_mode_records_a_shape_the_second_time_it_is_seen(self) -> None:
        runner = _Runner(mode="exact")

        first = runner.run(_inputs(1, 100), 4, "exact")
        second = runner.run(_inputs(1, 100), 4, "exact")
        third = runner.run(_inputs(1, 100), 4, "exact")

        assert first is None  # eager
        assert second is not None  # the recording forward is answered by the new graph
        assert third is not None
        assert runner.recorded == [(1, 100)]
        assert runner.replayed == [(1, 100), (1, 100)]
        assert runner.stats.eager == {"unseen": 1}
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 1)

    def test_bucketed_mode_records_once_per_bucket(self) -> None:
        runner = _Runner()

        for length in (100, 110, 128, 129):
            runner.run(_inputs(1, length), 4, "bucketed")

        assert runner.recorded == [(1, 128), (1, 160)]
        assert (runner.stats.recorded, runner.stats.replayed) == (2, 2)

    def test_label_counts_share_one_graph(self) -> None:
        runner = _Runner()

        answers = [runner.run(_inputs(3, 100), classes, "bucketed") for classes in range(1, 101)]

        assert runner.recorded == [(4, 128)]
        assert runner.head_classes == list(range(1, 101))  # the head scores each forward's own label slots
        assert all(answer is not None and answer.shape == (3, classes) for classes, answer in enumerate(answers, 1))

    def test_a_fixed_label_allocation_also_replays(self) -> None:
        runner = _Runner()

        runner.run(_inputs(1, 100), None, "bucketed")  # 'fixed' allocation: the model's own label count

        assert runner.head_classes == [None]

    def test_batch_sizes_share_their_bucket(self) -> None:
        runner = _Runner()

        for batch in (3, 4, 5, 6, 7, 8):
            runner.run(_inputs(batch, 100), 4, "bucketed")

        assert runner.recorded == [(4, 128), (8, 128)]

    def test_after_warm_up_every_bucketed_forward_replays_whatever_the_mix(self) -> None:
        # Shape cycling: random batch sizes, lengths and label counts for ten
        # simulated minutes, one forward per 50 ms.
        runner = _Runner()
        shapes = runner.shape_count
        rng = random.Random(0)  # noqa: S311 -- a reproducible traffic mix, not a secret
        for step in range(12_000):
            runner.now = step * 0.05
            runner.run(_inputs(rng.randint(1, 32), rng.randint(1, 512)), rng.randint(1, 100), "bucketed")

        assert len(runner.recorded) == len(set(runner.recorded)) == shapes  # each shape once: nothing evicted
        assert runner.graph_count == shapes
        replayed, eager = runner.stats.replayed, dict(runner.stats.eager)
        for _ in range(1000):
            runner.run(_inputs(rng.randint(1, 32), rng.randint(1, 512)), rng.randint(1, 100), "bucketed")
        new_eager = {reason: count - eager.get(reason, 0) for reason, count in runner.stats.eager.items()}
        # Only forwards past the token bound run eagerly; every other forward replays.
        assert {reason for reason, count in new_eager.items() if count} <= {"too_large"}
        assert runner.stats.replayed - replayed == 1000 - new_eager.get("too_large", 0) > 0

    def test_the_least_recently_used_graph_is_dropped_past_the_bound_in_exact_mode(self) -> None:
        runner = _Runner(mode="exact", max_graphs=2)

        for length in (32, 32, 64, 64, 32, 96, 96, 64, 64):
            runner.run(_inputs(1, length), 4, "exact")

        assert runner.graph_count == 2
        # 32 and 64 recorded; 32 used again; 96 drops 64, which is then seen and recorded again.
        assert runner.recorded == [(1, 32), (1, 64), (1, 96), (1, 64)]
        # A length's relative-position table lives as long as a graph of that length.
        assert set(runner._relative_pos) == {64, 96}

    def test_recording_is_rationed_in_time(self) -> None:
        runner = _Runner(_model(hidden_size=768))  # 71 shapes
        shapes = sorted(bucketed_shapes(512, 2048))

        # Every call is a new shape: sixteen record at once, then none while time stands still.
        answers = [runner.run(_inputs(batch, length), 4, "bucketed") for batch, length in shapes[:40]]
        assert len(runner.recorded) == 16
        assert answers.count(None) == 24  # the rest ran eagerly
        assert runner.stats.eager == {"recording_paused": 24}

        runner.now += 2.0  # one more recording every 2 seconds
        for batch, length in shapes[40:50]:
            runner.run(_inputs(batch, length), 4, "bucketed")
        assert len(runner.recorded) == 17

        runner.now += 3600.0  # an idle hour refills only the burst
        for batch, length in shapes[50:]:
            runner.run(_inputs(batch, length), 4, "bucketed")
        assert len(runner.recorded) == 17 + min(16, len(shapes) - 50)

    def test_one_recording_at_a_time_in_the_process(self) -> None:
        runner = _Runner()

        assert cuda_graphs_module._RECORDING_LOCK.acquire(blocking=False)  # another model is recording
        try:
            assert runner.run(_inputs(1, 100), 4, "bucketed") is None
        finally:
            cuda_graphs_module._RECORDING_LOCK.release()
        assert runner.recorded == []
        assert runner.stats.eager == {"busy": 1}
        assert runner.run(_inputs(1, 100), 4, "bucketed") is not None
        assert runner.recorded == [(1, 128)]
        assert not cuda_graphs_module._RECORDING_LOCK.locked()

    def test_a_bucketed_runner_over_its_budget_stops_recording_and_keeps_its_graphs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _Runner()
        runner.budget, runner.bytes_per_graph = 250, 100

        with caplog.at_level(logging.WARNING, logger=cuda_graphs_module.__name__):
            for length in (32, 64, 96, 128):
                runner.run(_inputs(1, length), 4, "bucketed")

        assert runner.recorded == [(1, 32), (1, 64), (1, 96)]  # the third went over; no more
        assert runner.graph_count == 3
        assert runner.stats.eager == {"budget_full": 1}
        assert "3 of 52 shapes recorded" in caplog.text
        runner.run(_inputs(1, 32), 4, "bucketed")
        assert runner.replayed[-1] == (1, 32)  # graphs it holds keep replaying
        runner.clear()  # an out-of-memory error or unload starts over
        runner.run(_inputs(1, 128), 4, "bucketed")
        assert runner.recorded[-1] == (1, 128)

    def test_an_exact_runner_over_its_budget_drops_its_graphs(self) -> None:
        runner = _Runner(mode="exact")
        runner.budget, runner.bytes_per_graph = 250, 100

        for length in (32, 32, 64, 64):
            runner.run(_inputs(1, length), 4, "exact")
        assert runner.graph_count == 2
        runner.run(_inputs(1, 96), 4, "exact")
        answer = runner.run(_inputs(1, 96), 4, "exact")  # 300 bytes: over the budget

        assert answer is not None  # the recording answered before its graphs were dropped
        assert runner.graph_count == 0
        assert runner._relative_pos == {}
        assert runner.stats.drops == 1
        assert not runner.disabled

    def test_the_budget_counts_the_tensors_graphs_keep(self) -> None:
        runner = _Runner()
        runner.budget = 999
        runner._relative_pos[32] = torch.zeros(1000, dtype=torch.uint8)  # a 1,000-byte table

        runner.run(_inputs(1, 32), 4, "bucketed")
        runner.run(_inputs(1, 64), 4, "bucketed")

        assert runner.recorded == [(1, 32)]  # the table put the first recording over

    def test_evicting_graphs_does_not_give_their_memory_back(self) -> None:
        # Evicted graphs leave their share of the shared pool behind, so their
        # memory still counts until the graphs are dropped.
        runner = _Runner(mode="exact", max_graphs=1)
        runner.budget, runner.bytes_per_graph = 250, 100

        for length in (32, 32, 64, 64):
            runner.run(_inputs(1, length), 4, "exact")
        assert runner.graph_count == 1
        runner.run(_inputs(1, 96), 4, "exact")
        runner.run(_inputs(1, 96), 4, "exact")

        assert runner.graph_count == 0

    def test_no_recording_while_device_memory_is_short(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.headroom = False

        assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # would record: runs eagerly
        assert runner.run(_inputs(1, 100), 4, "bucketed") is not None  # replays still run
        assert runner.recorded == [(1, 128)]
        assert runner.stats.eager == {"no_headroom": 1}

    def test_a_shape_that_fails_to_record_runs_eagerly_and_the_others_keep_replaying(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.capture_error = RuntimeError("operation not permitted when stream is capturing")

        with caplog.at_level(logging.WARNING, logger=cuda_graphs_module.__name__):
            assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # the adapter answers eagerly

        assert not cuda_graphs_module._RECORDING_LOCK.locked()
        assert "failed for batch 1, length 224" in caplog.text
        assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # not tried again
        assert runner.run(_inputs(1, 100), 4, "bucketed") is not None  # other graphs still replay
        assert runner.recorded == [(1, 128)]
        assert not runner.disabled
        assert runner.stats.recording_failures == 1
        assert runner.stats.eager == {"recording_failed": 1, "failed_shape": 1}

    def test_three_shapes_that_fail_to_record_turn_graphs_off(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 32), 4, "bucketed")
        runner.keep_failing = True
        runner.capture_error = RuntimeError("operation not permitted when stream is capturing")

        with caplog.at_level(logging.ERROR, logger=cuda_graphs_module.__name__):
            for length in (64, 96, 128):
                assert runner.run(_inputs(1, length), 4, "bucketed") is None

        assert runner.disabled
        assert runner.graph_count == 0
        assert "off for this process after 3 shapes failed to record" in caplog.text
        assert runner.run(_inputs(1, 32), 4, "bucketed") is None
        assert runner.stats.eager["disabled"] == 1

    def test_a_first_replay_that_fails_counts_as_a_failed_recording(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.replay_error = RuntimeError("CUDA error: an illegal memory access was encountered")

        assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # the adapter answers eagerly

        assert set(runner._graphs) == {(1, 128)}  # the new graph is not kept
        assert runner.stats.recording_failures == 1
        assert runner.stats.eager == {"recording_failed": 1}
        runner.replay_error = None
        assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # not recorded again
        assert runner.stats.eager["failed_shape"] == 1
        assert not cuda_graphs_module._RECORDING_LOCK.locked()

    @pytest.mark.parametrize("first_use", [True, False], ids=["recording-forward", "later-forward"])
    def test_a_scoring_head_error_fails_only_its_forward(self, first_use: bool) -> None:
        # The head reads the caller's inputs, so its errors are the caller's:
        # the forward fails as it would eagerly, and no shape loses its graph.
        runner = _Runner()
        runner.run(_inputs(1, 32), 4, "bucketed")
        shapes = [(1, 64), (1, 96), (1, 128), (1, 160)]
        if not first_use:
            for _, length in shapes:
                runner.run(_inputs(1, length), 4, "bucketed")
        runner.head_error = IndexError("index 4 is out of bounds for dimension 1 with size 4")

        for _, length in shapes:  # more than the three strikes that would turn graphs off
            with pytest.raises(IndexError):
                runner.run(_inputs(1, length), 4, "bucketed")

        assert runner.stats.recording_failures == 0
        assert runner._failed == set()
        assert not runner.disabled
        assert set(runner._graphs) == {(1, 32), *shapes}  # the graphs are kept
        assert not cuda_graphs_module._RECORDING_LOCK.locked()
        runner.head_error = None
        replayed = runner.stats.replayed
        for _, length in shapes:
            assert runner.run(_inputs(1, length), 4, "bucketed") is not None
        assert runner.stats.replayed == replayed + len(shapes)

    def test_a_first_replay_that_runs_out_of_memory_reaches_oom_recovery(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.replay_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            runner.run(_inputs(1, 200), 4, "bucketed")

        assert (1, 224) not in runner._graphs
        assert runner.stats.recording_failures == 0
        assert not cuda_graphs_module._RECORDING_LOCK.locked()

    def test_running_out_of_memory_while_recording_pauses_recording(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.capture_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

        assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # answered eagerly, not an error

        assert runner.graph_count == 0
        assert not runner.disabled
        assert runner.stats.drops == 1
        assert runner.stats.recording_failures == 0  # memory is not the shape's fault
        assert not cuda_graphs_module._RECORDING_LOCK.locked()
        # Recording pauses for a minute, then resumes.
        runner.now += 59.0
        assert runner.run(_inputs(1, 300), 4, "bucketed") is None
        runner.now += 2.0
        assert runner.run(_inputs(1, 300), 4, "bucketed") is not None
        assert runner.recorded[-1] == (1, 320)

    @pytest.mark.parametrize(
        "inputs",
        [
            {**_inputs(1, 100), "class_input_ids": torch.ones(1, 4, dtype=torch.long)},
            {"input_ids": torch.ones(1, 100, dtype=torch.long)},  # the head needs the mask
        ],
        ids=["extra-input", "no-mask"],
    )
    def test_unexpected_inputs_run_eagerly(self, inputs: dict[str, torch.Tensor]) -> None:
        runner = _Runner()

        assert runner.run(inputs, 4, "bucketed") is None
        assert runner.recorded == []
        assert runner.stats.eager == {"unsupported_inputs": 1}

    def test_every_forward_counts_once(self) -> None:
        runner = _Runner()

        runner.run(_inputs(1, 100), 4, "bucketed")  # recorded
        runner.run(_inputs(1, 100), 8, "bucketed")  # replayed
        runner.run(_inputs(4, 512), 8, "bucketed")  # too large
        runner.run(_inputs(4, 512), 8, "off")  # opted out: not counted

        assert (runner.stats.recorded, runner.stats.replayed, runner.stats.eager) == (1, 1, {"too_large": 1})
        assert runner.stats.forwards == 3

    def test_the_runner_logs_its_counts_every_ten_minutes(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner(name="knowledgator/gliclass-large-v1.0")

        with caplog.at_level(logging.INFO, logger=cuda_graphs_module.__name__):
            for _ in range(4):
                runner.run(_inputs(1, 100), 4, "bucketed")
            assert "forwards in the last" not in caplog.text
            runner.now += 600.0
            runner.run(_inputs(4, 512), 4, "bucketed")

        assert (
            "GLiClass CUDA graphs for knowledgator/gliclass-large-v1.0: 5 forwards in the last 600 s, 60.0% replayed"
            in (caplog.text)
        )


class TestRecordingHooks:
    def test_hooks_leave_eager_forwards_alone(self) -> None:
        model = _model(segment_embeddings=True)
        _Runner(model)
        encoder = model.model.encoder_model.encoder
        hidden = torch.zeros(1, 7, 4)

        assert encoder.get_rel_pos(hidden) == ("built", 7)
        assert model.model._create_segment_ids(torch.ones(1, 3, dtype=torch.long)) == "eager"
        assert model.model.process_encoder_output("ids", "mask", hidden, None, 3) == ("head", hidden)

    def test_while_recording_the_encoder_gets_the_precomputed_table(self) -> None:
        model = _model(segment_embeddings=True)
        runner = _Runner(model)
        encoder = model.model.encoder_model.encoder
        table = torch.arange(4)

        with runner._recording_on_this_thread(table):
            assert encoder.get_rel_pos(torch.zeros(1, 7, 4)) is table
            assert encoder.get_rel_pos(torch.zeros(1, 7, 4), None, "given") == ("built", 7)
            ids = torch.tensor([[0, 8, 5, 5, 2]])
            assert torch.equal(model.model._create_segment_ids(ids), torch.tensor([[0, 1, 1, 1, 1]]))
        assert encoder.get_rel_pos(torch.zeros(1, 7, 4)) == ("built", 7)

    def test_while_recording_the_forward_stops_at_the_hidden_states(self) -> None:
        model = _model()
        runner = _Runner(model)
        hidden = torch.zeros(2, 7, 4)

        with runner._recording_on_this_thread(None):
            logits, *rest = model.model.process_encoder_output("ids", "mask", hidden, None, 3)

        assert logits is hidden
        assert rest == [None, None, None]

    def test_recording_diverts_only_the_recording_thread(self) -> None:
        # Another thread's eager forward of the same model, while a graph
        # records, must get logits, not the recording's hidden states.
        model = _model(segment_embeddings=True)
        runner = _Runner(model)
        encoder = model.model.encoder_model.encoder
        hidden = torch.zeros(2, 7, 4)
        seen: dict[str, Any] = {}

        def eager_forward() -> None:
            seen["head"] = model.model.process_encoder_output("ids", "mask", hidden, None, 3)
            seen["table"] = encoder.get_rel_pos(hidden)
            seen["segments"] = model.model._create_segment_ids(torch.ones(1, 3, dtype=torch.long))

        with runner._recording_on_this_thread(torch.arange(4)):
            other = threading.Thread(target=eager_forward)
            other.start()
            other.join()
            assert model.model.process_encoder_output("ids", "mask", hidden, None, 3)[0] is hidden

        assert seen == {"head": ("head", hidden), "table": ("built", 7), "segments": "eager"}

    def test_the_recording_state_is_cleared_when_recording_fails(self) -> None:
        runner = _Runner()

        with pytest.raises(RuntimeError), runner._recording_on_this_thread(torch.arange(4)):
            raise RuntimeError("capture failed")

        assert not runner._recording_here()


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 5, 8, 5, 5, 2]],  # text token, no examples
        [[0, 5, 8, 5, 9, 5, 2]],  # text then an example
        [[0, 9, 9, 8, 5, 2]],  # examples first
        [[0, 5, 5, 5, 5, 2]],  # neither
        [[0, 8, 5, 9, 5, 2], [0, 5, 5, 8, 5, 2]],  # rows differ
    ],
)
def test_segment_ids_are_the_libraries_own(rows: list[list[int]]) -> None:
    config = SimpleNamespace(example_token_index=9, text_token_index=8)
    input_ids = torch.tensor(rows)

    expected = GLiClassUniEncoder._create_segment_ids(SimpleNamespace(config=config), input_ids)  # ty:ignore[invalid-argument-type]

    assert torch.equal(segment_ids(input_ids, config), expected)


class _Pipe:
    """A gliclass pipe whose eager forward fails with ``error``, or answers with zeros."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def _resolve_max_num_classes(self, labels: list[str], same_labels: bool) -> int:
        return len(labels)

    def model(self, **inputs: Any) -> Any:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(logits=torch.zeros(inputs["input_ids"].shape[0], inputs["max_num_classes"]))


def _adapter_with(runner: _Runner) -> GLiClassAdapter:
    adapter = GLiClassAdapter("tiny", max_seq_length=512, cuda_graphs="bucketed")
    adapter._graphs = runner
    return adapter


_OOM = torch.cuda.OutOfMemoryError("CUDA out of memory")


def test_an_eager_out_of_memory_drops_the_graphs() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    assert runner.graph_count == 1
    runner._recording_credit = 0.0  # the next shape is not recorded and runs eagerly

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _adapter_with(runner)._forward(_Pipe(_OOM), _inputs(1, 300), ["a", "b"], same_labels=True, graphs="bucketed")

    assert runner.graph_count == 0
    assert runner.recorded == [(1, 128)]
    assert not runner.disabled


def test_an_eager_error_worded_as_out_of_memory_drops_the_graphs() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    pipe = _Pipe(RuntimeError("CUDA error: out of memory"))

    with pytest.raises(RuntimeError):
        _adapter_with(runner)._forward(pipe, _inputs(1, 100), ["a", "b"], same_labels=True, graphs="off")

    assert runner.graph_count == 0


def test_other_eager_errors_keep_the_graphs() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    pipe = _Pipe(ValueError("bad input"))

    with pytest.raises(ValueError, match="bad input"):
        _adapter_with(runner)._forward(pipe, _inputs(1, 100), ["a", "b"], same_labels=True, graphs="off")

    assert runner.graph_count == 1


def test_a_recording_out_of_memory_answers_eagerly() -> None:
    runner = _Runner()
    runner.capture_error = _OOM

    logits = _adapter_with(runner)._forward(_Pipe(), _inputs(1, 100), ["a", "b"], same_labels=True, graphs="bucketed")

    assert torch.equal(logits, torch.zeros(1, 2))  # the eager forward's answer
    assert runner.graph_count == 0
    assert not runner.disabled


def test_a_recording_out_of_memory_that_persists_reaches_oom_recovery() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    runner.capture_error = _OOM

    with pytest.raises(torch.cuda.OutOfMemoryError):  # the eager forward also runs out
        _adapter_with(runner)._forward(_Pipe(_OOM), _inputs(1, 300), ["a", "b"], same_labels=True, graphs="bucketed")

    assert runner.graph_count == 0
    assert not runner.disabled


def test_an_out_of_memory_error_in_a_replay_drops_the_graphs() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    runner.replay_error = _OOM

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _adapter_with(runner)._forward(_Pipe(), _inputs(1, 100), ["a", "b"], same_labels=True, graphs="bucketed")

    assert runner.graph_count == 0


def test_the_adapter_passes_its_label_slots_to_the_head() -> None:
    runner = _Runner()

    logits = _adapter_with(runner)._forward(
        _Pipe(), _inputs(3, 100), ["a", "b", "c"], same_labels=True, graphs="bucketed"
    )

    assert logits.shape == (3, 3)
    assert runner.head_classes == [3]
    assert runner.recorded == [(4, 128)]


class TestOperatorSetting:
    """Graphs are chosen when the model loads; a request can only opt out."""

    @pytest.mark.parametrize("mode", ["off", "exact", "bucketed"])
    def test_requests_use_the_load_time_mode(self, mode: str) -> None:
        adapter = GLiClassAdapter("tiny", cuda_graphs=mode)

        assert adapter._request_cuda_graphs({}) == mode
        assert adapter._request_cuda_graphs({"cuda_graphs": "off"}) == "off"

    def test_a_request_may_send_false_for_off(self) -> None:
        # A profile's runtime options read an unquoted YAML ``off`` as False.
        assert GLiClassAdapter("tiny", cuda_graphs="bucketed")._request_cuda_graphs({"cuda_graphs": False}) == "off"

    @pytest.mark.parametrize("value", ["exact", "bucketed", "on", "Off", True, None, 0, 1])
    def test_a_request_can_only_turn_graphs_off(self, value: object) -> None:
        adapter = GLiClassAdapter("tiny", cuda_graphs="bucketed")

        with pytest.raises(InvalidInputError, match="accepts only 'off'"):
            adapter._request_cuda_graphs({"cuda_graphs": value})

    @pytest.mark.parametrize(("text", "mode"), [("off", "off"), ('"off"', "off"), ("bucketed", "bucketed")])
    def test_profiles_may_write_off_unquoted(self, text: str, mode: str) -> None:
        loadtime = yaml.safe_load(f"cuda_graphs: {text}")  # YAML reads unquoted off as False

        assert GLiClassAdapter("tiny", **loadtime)._cuda_graphs == mode

    @pytest.mark.parametrize("value", ["on", "Exact", "", None, True, 0])
    def test_unknown_load_time_modes_fail_the_load(self, value: Any) -> None:
        with pytest.raises(ValueError, match="cuda_graphs must be 'off', 'exact' or 'bucketed'"):
            GLiClassAdapter("tiny", cuda_graphs=value)

    def test_profiles_may_set_it_at_load(self) -> None:
        reject_unknown_loadtime_options(GLiClassAdapter, {"cuda_graphs": "bucketed"}, model_name="tiny")

    def test_a_request_that_opts_out_runs_eagerly(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 2, "bucketed")
        replays = len(runner.replayed)

        logits = _adapter_with(runner)._forward(_Pipe(), _inputs(1, 100), ["a", "b"], same_labels=True, graphs="off")

        assert torch.equal(logits, torch.zeros(1, 2))
        assert len(runner.replayed) == replays


def test_shipped_profiles_enable_bucketed_graphs_only_where_measured() -> None:
    configs = load_model_configs(_MODELS_DIR)
    # Named profiles (``model:profile``) inherit the default profile's load-time options.
    modes = {
        name.split(":")[0]: config.resolve_profile("default").loadtime.get("cuda_graphs", "off")
        for name, config in configs.items()
        if config.resolve_profile("default").adapter_path.endswith(":GLiClassAdapter")
    }

    assert {name for name, mode in modes.items() if mode != "off"} == _BUCKETED_BY_DEFAULT
    assert {modes[name] for name in _BUCKETED_BY_DEFAULT} == {"bucketed"}
    for name in _BUCKETED_BY_DEFAULT:
        adapter = GLiClassAdapter(**_build_adapter_kwargs(configs[name], "float16"))
        assert adapter._cuda_graphs == "bucketed"
        assert adapter._request_cuda_graphs({"cuda_graphs": "off"}) == "off"
