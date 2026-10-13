"""GLiNER2.5-Decide CUDA graph policy: which calls replay a graph, which record one, which run eagerly.

These run without a GPU: recording and replay are replaced by fakes. Scores
from real graphs were checked against eager scores on an L4 and an RTX PRO 6000
(see ``decide_graphs.py``).
"""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from sie_server.adapters._cuda_graphs import RECORDING_LOCK
from sie_server.adapters.gliner2 import decide_graphs as graphs_module
from sie_server.adapters.gliner2.decide import GLiNER2DecideAdapter
from sie_server.adapters.gliner2.decide_graphs import (
    DecideGraphRunner,
    batch_sizes,
    bucket_length,
    graph_shape,
    shapes,
    unsupported_reason,
)
from sie_server.core.loader import _build_adapter_kwargs, load_model_configs, reject_unknown_loadtime_options

from .test_gliner2_decide import FakeEncoder, expected_logits, make_adapter

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# The shipped profile that loads with graphs: on the 400 messages of the
# classify study its graphs moved no probability by more than 0.0014 against
# eager execution and changed no top label, on an L4 and on an RTX PRO 6000.
_BUCKETED_BY_DEFAULT = {"fastino/GLiNER2.5-Decide"}
_DEVICE = torch.device("cpu")


class _Layers:
    def __init__(self) -> None:
        self.built: list[int] = []

    def get_rel_pos(self, hidden_states: torch.Tensor, query_states: Any = None, relative_pos: Any = None) -> Any:
        self.built.append(hidden_states.shape[-2])
        return ("built", hidden_states.shape[-2])


def _encoder(encoder_type: str = "deberta-v2", hidden_size: int = 1024, hook: bool = True) -> Any:
    layers = _Layers() if hook else SimpleNamespace()
    return SimpleNamespace(config=SimpleNamespace(model_type=encoder_type, hidden_size=hidden_size), encoder=layers)


class _Runner(DecideGraphRunner):
    """Records and replays graphs with fakes; the runner's own code decides what happens when.

    A replay returns zeros for the call's rows at the graph's length, or raises
    ``replay_error``. A recording whose capture fails (``capture_error``)
    raises that error. Time stands still unless a test moves ``now``; the
    device always has memory to spare unless a test clears ``headroom`` or sets
    a ``budget``. The host launches a forward in ``host_launch_ms`` and the GPU
    runs a graph in ``replay_ms_per_token`` for each of its padded tokens.
    """

    # Whether ``capture_error`` fails every recording, not just the next one.
    keep_failing = False
    # Device memory a recording adds to the pool the graphs share.
    pool_growth = 0

    def __init__(self, encoder: Any = None, **kwargs: Any) -> None:
        self.now = 0.0
        kwargs.setdefault("window", 512)
        super().__init__(encoder or _encoder(), clock=lambda: self.now, **kwargs)
        self.recorded: list[tuple[int, int]] = []
        self.replayed: list[tuple[int, int]] = []
        self.capture_error: Exception | None = None
        self.replay_error: Exception | None = None
        self.headroom = True
        self.budget = 2**40
        self.bytes_per_graph = 0
        self.host_launch_ms = 30.0
        self.replay_ms_per_token = 0.0
        self.launches_timed = 0
        self.launch_forwards: list[int] = []
        self.free = 2**40

    def _free_memory(self, device: torch.device) -> int:
        return self.free

    def _time_launch(self, key: Any, entry: Any, forwards: int = 3) -> float:
        self.launches_timed += 1
        self.launch_forwards.append(forwards)
        return self.host_launch_ms

    def _time_replay(self, entry: Any) -> float:
        return self.replay_ms_per_token * entry.input_ids.shape[0] * entry.length

    def _has_headroom(self, device: torch.device) -> bool:
        return self.headroom

    def _memory_budget(self, device: torch.device) -> int:
        return self.budget

    def _record(self, key: Any, device: torch.device) -> Any:
        if self.capture_error is not None:
            error, self.capture_error = self.capture_error, (self.capture_error if self.keep_failing else None)
            raise error
        self.recorded.append(key)
        self._relative_pos.setdefault(key[1], torch.zeros(1))
        # Static inputs with a row per padded row and no memory to count.
        empty = torch.zeros(key[0], 0, dtype=torch.long)
        self.free -= self.pool_growth
        return SimpleNamespace(
            graph=None, input_ids=empty, attention_mask=empty, device_bytes=self.bytes_per_graph, length=key[1]
        )

    def _replay(self, key: Any, entry: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.replay_error is not None:
            raise self.replay_error
        self.replayed.append(key)
        return torch.zeros(input_ids.shape[0], key[1], 2)

    def offer(self, rows: int, length: int) -> torch.Tensor | None:
        """Offer the runner a call of ``rows`` rows of ``length`` tokens, as the adapter does."""
        key = self.key(rows, length)
        if key is None:
            return None
        ids = torch.ones(rows, length, dtype=torch.long)
        return self.run(ids, ids, key)


class TestSupport:
    def test_a_deberta_encoder_on_cuda_is_supported(self) -> None:
        assert unsupported_reason(_encoder(), "cuda:0", 512) is None

    @pytest.mark.parametrize(
        ("encoder", "device", "window", "reason"),
        [
            (_encoder(), "cpu", 512, "need a CUDA device"),
            (_encoder(), "mps", 512, "need a CUDA device"),
            (_encoder(encoder_type="modernbert"), "cuda", 512, "modernbert encoder is not supported"),
            (_encoder(hook=False), "cuda", 512, "no relative-position hook"),
            (_encoder(), "cuda", 4096, "does not fit a graph"),
        ],
    )
    def test_other_encoders_devices_and_windows_run_eagerly(
        self, encoder: Any, device: str, window: int, reason: str
    ) -> None:
        message = unsupported_reason(encoder, device, window)

        assert message is not None
        assert reason in message


class TestShapes:
    @pytest.mark.parametrize(
        ("length", "padded"),
        [(1, 16), (16, 16), (17, 32), (47, 48), (62, 64), (65, 80), (128, 128), (129, 160), (500, 512), (512, 512)],
    )
    def test_lengths_round_up_to_their_bucket(self, length: int, padded: int) -> None:
        assert bucket_length(length, 512) == padded

    def test_a_bucket_never_passes_the_window(self) -> None:
        assert bucket_length(300, 300) == 300
        assert bucket_length(290, 300) == 300

    @pytest.mark.parametrize(
        ("length", "largest"), [(16, 32), (64, 32), (80, 24), (128, 16), (160, 12), (256, 8), (320, 6), (512, 4)]
    )
    def test_a_graph_holds_at_most_2048_tokens_and_32_rows(self, length: int, largest: int) -> None:
        sizes = batch_sizes(length)

        assert sizes[-1] == largest
        assert all(size * length <= 2048 for size in sizes)

    @pytest.mark.parametrize(
        ("rows", "batch"),
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (8, 8), (9, 12), (13, 16), (17, 24), (25, 32), (32, 32)],
    )
    def test_a_batch_pads_up_to_the_next_batch_size(self, rows: int, batch: int) -> None:
        assert graph_shape(rows, 60, 512) == (batch, 64)

    def test_a_call_that_fits_no_graph_runs_eagerly(self) -> None:
        assert graph_shape(33, 60, 512) is None  # more rows than a graph holds
        assert graph_shape(17, 100, 512) is None  # 24 rows of 112 tokens pass the token bound
        assert graph_shape(5, 500, 512) is None
        assert graph_shape(4, 500, 512) == (4, 512)
        assert graph_shape(1, 513, 512) is None  # past the window
        assert graph_shape(0, 60, 512) is None

    def test_every_call_that_replays_maps_into_a_small_fixed_set(self) -> None:
        fixed = shapes(512)

        assert len(fixed) == 132
        rng = random.Random(7)  # noqa: S311 -- reproducible shapes, not security
        for _ in range(5000):
            rows, longest = rng.randint(1, 80), rng.randint(1, 512)
            key = graph_shape(rows, longest, 512)
            if key is None:
                # Not even the batch that just holds the rows fits the token bound.
                assert rows > batch_sizes(bucket_length(longest, 512))[-1]
            else:
                assert key in fixed
                assert key[0] >= rows
                assert key[1] >= longest
                assert key[0] * key[1] <= 2048

    def test_under_a_third_of_a_padded_batch_is_padding(self) -> None:
        for rows in range(1, 33):
            key = graph_shape(rows, 16, 512)
            assert key is not None
            assert 3 * (key[0] - rows) < key[0]


class TestRecordingPolicy:
    def test_a_shape_is_recorded_once_and_replayed_from_then_on(self) -> None:
        runner = _Runner()

        for _ in range(3):
            assert runner.offer(1, 50) is not None

        assert runner.recorded == [(1, 64)]
        assert runner.replayed == [(1, 64)] * 3  # the recording forward is answered by the first replay
        assert (runner.stats.recorded, runner.stats.replayed) == (1, 2)

    def test_lengths_and_batches_share_their_bucket(self) -> None:
        runner = _Runner()

        for rows, length in [(1, 49), (1, 64), (5, 60), (6, 51)]:
            runner.offer(rows, length)

        assert runner.recorded == [(1, 64), (6, 64)]

    def test_the_hidden_states_are_those_of_the_calls_rows(self) -> None:
        hidden = _Runner().offer(5, 50)

        assert hidden is not None
        assert hidden.shape == (5, 64, 2)

    def test_a_call_that_fits_no_graph_is_not_offered(self) -> None:
        runner = _Runner()

        assert runner.offer(33, 50) is None
        assert runner.stats.forwards == 0

    def test_after_warm_up_every_call_replays_whatever_the_mix(self) -> None:
        runner = _Runner()
        rng = random.Random(3)  # noqa: S311 -- reproducible shapes, not security
        calls = [(rng.randint(1, 8), rng.randint(40, 80)) for _ in range(400)]
        for rows, length in calls:  # warm-up: the clock moves, so the recording budget refills
            runner.now += 2.0
            runner.offer(rows, length)
        recorded = len(runner.recorded)

        for rows, length in calls:
            assert runner.offer(rows, length) is not None

        assert len(runner.recorded) == recorded <= 6 * 3  # six batch sizes, three length buckets
        assert runner.stats.eager == {}

    def test_recording_is_rationed_in_time(self) -> None:
        runner = _Runner()

        results = [runner.offer(1, 16 * (step + 1)) for step in range(8)] + [
            runner.offer(2, 16 * (step + 1)) for step in range(8)
        ]
        assert all(result is not None for result in results)  # a burst of 16 recordings
        assert runner.offer(3, 16) is None  # the 17th waits
        assert runner.stats.eager == {"recording_paused": 1}

        runner.now += 2.0  # one more recording every 2 seconds
        assert runner.offer(3, 16) is not None
        assert runner.offer(4, 16) is None

    def test_one_recording_at_a_time_in_the_process(self) -> None:
        runner = _Runner()

        with RECORDING_LOCK:  # another model's runner is recording
            assert runner.offer(1, 50) is None
        assert runner.stats.eager == {"busy": 1}

        assert runner.offer(1, 50) is not None

    def test_a_runner_over_its_budget_stops_recording_and_keeps_its_graphs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _Runner()
        runner.bytes_per_graph, runner.budget = 100, 250

        with caplog.at_level(logging.WARNING):
            for rows in (1, 2, 3, 4):
                runner.offer(rows, 50)

        assert runner.recorded == [(1, 64), (2, 64), (3, 64)]
        assert runner.graph_count == 3
        assert runner.offer(1, 50) is not None  # the recorded shapes keep replaying
        assert runner.stats.eager == {"budget_full": 1}
        assert "their memory budget, with 3 of 132 shapes recorded" in caplog.text

    def test_the_budget_counts_the_tensors_graphs_keep(self) -> None:
        runner = _Runner()
        runner.offer(1, 50)
        assert runner.held_bytes == torch.zeros(1).nbytes  # the fake relative-position table

        runner._hidden = torch.zeros(1000, dtype=torch.float16)
        assert runner.held_bytes == torch.zeros(1).nbytes + 2000

    def test_no_recording_while_device_memory_is_short(self) -> None:
        runner = _Runner()
        runner.headroom = False

        assert runner.offer(1, 50) is None
        assert runner.stats.eager == {"no_headroom": 1}

        runner.headroom = True
        assert runner.offer(1, 50) is not None

    def test_a_shape_that_fails_to_record_runs_eagerly_and_the_others_keep_replaying(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _Runner()
        runner.offer(1, 50)
        runner.capture_error = RuntimeError("cannot capture")

        with caplog.at_level(logging.WARNING):
            assert runner.offer(2, 50) is None
        assert runner.offer(2, 50) is None  # not tried again
        assert runner.offer(1, 50) is not None

        assert runner.stats.eager == {"recording_failed": 1, "failed_shape": 1}
        assert runner.stats.recording_failures == 1
        assert "failed for batch 2, length 64" in caplog.text
        assert not runner.disabled
        assert not RECORDING_LOCK.locked()

    def test_three_shapes_that_fail_to_record_turn_graphs_off(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner()
        runner.offer(1, 50)
        runner.capture_error, runner.keep_failing = RuntimeError("cannot capture"), True

        with caplog.at_level(logging.ERROR):
            for rows in (2, 3, 4):
                assert runner.offer(rows, 50) is None

        assert runner.disabled
        assert runner.graph_count == 0
        assert runner.offer(1, 50) is None
        assert runner.stats.eager == {"recording_failed": 3, "disabled": 1}
        assert "are off for this process after 3 shapes failed to record" in caplog.text

    def test_a_first_replay_that_fails_counts_as_a_failed_recording(self) -> None:
        runner = _Runner()
        runner.replay_error = RuntimeError("replay failed")

        assert runner.offer(1, 50) is None

        assert runner.graph_count == 0
        assert runner.stats.recorded == 0
        assert runner.stats.recording_failures == 1
        runner.replay_error = None
        assert runner.offer(1, 50) is None  # that shape is not recorded again
        assert runner.stats.eager == {"recording_failed": 1, "failed_shape": 1}

    def test_a_first_replay_that_runs_out_of_memory_reaches_oom_recovery_and_pauses_recording(self) -> None:
        runner = _Runner()
        runner.offer(1, 50)
        runner.replay_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            runner.offer(2, 50)

        assert runner.graph_count == 0  # every graph dropped
        assert runner.stats.drops == 1
        assert runner.stats.recording_failures == 0  # memory is not the shape's fault
        assert not RECORDING_LOCK.locked()
        runner.replay_error = None
        assert runner.offer(2, 50) is None
        assert runner.stats.eager == {"recording_paused": 1}
        runner.now += 60.0
        assert runner.offer(2, 50) is not None

    def test_running_out_of_memory_while_recording_pauses_recording(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner()
        runner.offer(1, 50)
        runner.capture_error = graphs_module._RecordingOutOfMemoryError("CUDA out of memory")

        with caplog.at_level(logging.INFO):
            assert runner.offer(2, 50) is None

        assert runner.graph_count == 0
        assert "ran out of memory; graphs dropped, recording paused for 60 s" in caplog.text
        runner.now += 59.0
        assert runner.offer(2, 50) is None
        runner.now += 1.0
        assert runner.offer(2, 50) is not None

    def test_every_call_counts_once(self) -> None:
        runner = _Runner()
        runner.capture_error = RuntimeError("cannot capture")

        for rows in (1, 1, 2, 2, 2):
            runner.offer(rows, 50)

        stats = runner.stats
        assert stats.forwards == 5
        assert (stats.recorded, stats.replayed, stats.eager) == (1, 2, {"recording_failed": 1, "failed_shape": 1})

    def test_the_runner_logs_its_counts_every_ten_minutes(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner(name="fastino/GLiNER2.5-Decide")

        with caplog.at_level(logging.INFO):
            runner.offer(1, 50)
            assert "forwards in the last" not in caplog.text
            runner.now += 600.0
            runner.offer(1, 50)
            runner.offer(1, 50)

        assert caplog.text.count("forwards in the last") == 1
        assert "fastino/GLiNER2.5-Decide: 2 forwards in the last 600 s, 50.0% replayed" in caplog.text
        assert "1 graphs" in caplog.text

    def test_clear_drops_every_graph_and_what_it_kept(self) -> None:
        runner = _Runner()
        runner.bytes_per_graph, runner.budget = 100, 150
        runner.offer(1, 50)
        runner.offer(2, 50)  # over the budget: recording stops

        runner.clear()

        assert (runner.graph_count, runner.held_bytes) == (0, 0)
        assert runner.offer(3, 50) is not None  # and may start again


class TestGraphsThatDoNotPay:
    """A graph is kept only if the GPU replays it no slower than the host launches a forward."""

    def test_a_graph_slower_than_a_launch_is_not_kept_and_its_shape_runs_eagerly(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03  # an L4: about 1,000 tokens in a launch

        assert runner.offer(8, 60) is not None  # 512 tokens: 15 ms
        assert runner.offer(16, 60) is None  # 1,024 tokens: 31 ms
        assert runner.offer(16, 60) is None  # not recorded again

        assert runner.recorded == [(8, 64), (16, 64)]
        assert runner.graph_count == 1
        assert runner.stats.eager == {"not_faster": 2}
        assert runner.stats.recording_failures == 0
        assert not runner.disabled
        assert not RECORDING_LOCK.locked()

    def test_a_faster_gpu_keeps_more_shapes(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 15.0, 0.006  # an RTX PRO 6000

        assert runner.offer(16, 60) is not None
        assert runner.offer(32, 60) is not None  # 2,048 tokens: 12 ms
        assert runner.stats.eager == {}

    def test_a_replay_as_long_as_a_launch_keeps_its_graph(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 32.0, 0.5

        assert runner.offer(1, 60) is not None  # 64 tokens: 32 ms

    def test_the_launch_is_timed_with_every_recording_and_its_least_kept(self) -> None:
        runner = _Runner()
        assert runner.launch_ms is None

        runner.offer(1, 60)
        runner.host_launch_ms = 45.0  # the CPU was busy this time
        runner.offer(2, 60)
        runner.host_launch_ms = 28.0
        runner.offer(3, 60)
        runner.offer(3, 60)  # a replay times nothing

        assert runner.launch_forwards == [3, 1, 1]  # three one-row forwards the first time, one after
        assert runner.launch_ms == 28.0

    def test_shapes_no_smaller_than_a_slow_one_are_not_recorded(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03
        runner.offer(16, 60)  # slower than a launch
        recorded = list(runner.recorded)

        assert runner.offer(16, 70) is None  # as many rows, longer
        assert runner.offer(24, 60) is None  # more rows, as long
        assert runner.offer(32, 10) is not None  # more rows but shorter (512 tokens): timed for itself
        assert runner.offer(12, 70) is not None  # fewer rows (960 tokens): timed for itself

        assert runner.recorded == [*recorded, (32, 16), (12, 80)]
        assert runner.stats.eager == {"not_faster": 3}

    def test_what_the_runner_learned_about_the_host_survives_dropping_the_graphs(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03
        runner.offer(8, 60)
        runner.offer(16, 60)

        runner.clear()

        assert runner.offer(16, 60) is None
        assert runner.recorded.count((16, 64)) == 1  # not timed again yet
        assert runner.offer(8, 60) is not None
        assert runner.launch_ms == 30.0

    def test_memory_a_dropped_recording_left_in_the_pool_counts_against_the_budget(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03
        runner.pool_growth = 100

        runner.offer(16, 60)

        assert runner.graph_count == 0
        assert runner.held_bytes == 100 + torch.zeros(1).nbytes  # and the length's relative-position table

    def test_the_summary_counts_the_shapes_that_do_not_pay(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = _Runner(name="fastino/GLiNER2.5-Decide")
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03

        with caplog.at_level(logging.INFO):
            runner.offer(16, 60)
            runner.now += 600.0
            runner.offer(16, 60)  # timed again, slower again

        assert "this host launches a forward in 30.0 ms" in caplog.text
        assert "1 shapes not faster than a launch" in caplog.text

    def test_a_shape_found_slower_is_timed_again_after_a_minute_and_kept_once_it_pays(self) -> None:
        # Engines that warm up together on one GPU time their first graph while sharing it.
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 9.0, 1.0

        assert runner.offer(1, 10) is None  # the smallest shape: 16 ms on a shared GPU
        assert runner.offer(8, 60) is None  # and with it every shape, for now
        runner.now += 59.0
        assert runner.offer(1, 10) is None
        assert runner.recorded == [(1, 16)]

        runner.now += 1.0
        runner.replay_ms_per_token = 0.006  # the GPU is this engine's own again
        assert runner.offer(1, 10) is not None
        assert runner.offer(8, 60) is not None
        assert runner.recorded == [(1, 16), (1, 16), (8, 64)]
        assert runner.stats.eager == {"not_faster": 3}
        assert runner.stats.recording_failures == 0

    def test_the_wait_doubles_each_time_a_shape_is_found_slower_again(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03
        timed_at = []

        for second in range(0, 1000, 10):
            runner.now = float(second)
            before = len(runner.recorded)
            runner.offer(16, 60)
            if len(runner.recorded) > before:
                timed_at.append(second)

        assert timed_at == [0, 60, 180, 420, 900]  # after 60 s, then 120, 240, 480

    def test_the_wait_stops_growing_at_an_hour(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 30.0, 0.03

        for _ in range(12):
            runner.offer(16, 60)
            runner.now = runner._slow[16, 64][0]

        assert runner._slow[16, 64][1] == 3600.0

    def test_a_graph_kept_against_a_launch_time_that_proves_shorter_is_dropped(self) -> None:
        runner = _Runner()
        runner.host_launch_ms, runner.replay_ms_per_token = 60.0, 0.03  # the CPU was busy at the first timing
        assert runner.offer(16, 60) is not None  # 31 ms, kept against 60 ms
        runner.host_launch_ms = 30.0
        assert runner.offer(1, 60) is not None  # this recording times the launch again: 30 ms

        assert runner.offer(16, 60) is None

        assert runner.graph_count == 1
        assert runner.stats.eager == {"not_faster": 1}
        assert runner.offer(16, 60) is None  # and waits to be timed again like any shape found slower
        assert runner.recorded.count((16, 64)) == 1


class TestRecordingHook:
    def test_the_hook_leaves_eager_forwards_alone(self) -> None:
        encoder = _encoder()
        DecideGraphRunner(encoder, window=512)

        assert encoder.encoder.get_rel_pos(torch.zeros(1, 7, 1)) == ("built", 7)

    def test_while_recording_the_encoder_gets_the_precomputed_table(self) -> None:
        encoder = _encoder()
        runner = DecideGraphRunner(encoder, window=512)

        with runner._recording_on_this_thread("precomputed"):
            assert encoder.encoder.get_rel_pos(torch.zeros(1, 7, 1)) == "precomputed"
            # A caller that brings its own table keeps it.
            assert encoder.encoder.get_rel_pos(torch.zeros(1, 7, 1), None, "own") == ("built", 7)
        assert encoder.encoder.get_rel_pos(torch.zeros(1, 7, 1)) == ("built", 7)

    def test_recording_diverts_only_the_recording_thread(self) -> None:
        encoder = _encoder()
        runner = DecideGraphRunner(encoder, window=512)
        seen: list[Any] = []

        with runner._recording_on_this_thread("precomputed"):
            thread = threading.Thread(target=lambda: seen.append(encoder.encoder.get_rel_pos(torch.zeros(1, 9, 1))))
            thread.start()
            thread.join()

        assert seen == [("built", 9)]

    def test_each_thread_warms_the_recording_stream_up_for_itself(self) -> None:
        # The engine loads and warms a model up on one thread and serves it on another; a thread that
        # records without having used the runner's stream fails in cublasCreate.
        runner = DecideGraphRunner(_encoder(), window=512)
        runner._local.warmed = True  # what a recording on this thread leaves
        seen: list[bool] = []

        thread = threading.Thread(target=lambda: seen.append(getattr(runner._local, "warmed", False)))
        thread.start()
        thread.join()

        assert seen == [False]

    def test_the_recording_state_is_cleared_when_recording_fails(self) -> None:
        encoder = _encoder()
        runner = DecideGraphRunner(encoder, window=512)

        with pytest.raises(RuntimeError, match="boom"), runner._recording_on_this_thread("precomputed"):
            raise RuntimeError("boom")

        assert encoder.encoder.get_rel_pos(torch.zeros(1, 7, 1)) == ("built", 7)


class _PaddingRunner(_Runner):
    """Replays by running the adapter's fake encoder on the call padded to its graph's shape."""

    def _replay(self, key: Any, entry: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.replay_error is not None:
            raise self.replay_error
        self.replayed.append(key)
        rows, length = input_ids.shape
        ids = torch.zeros(key, dtype=torch.long)
        mask = torch.zeros(key, dtype=torch.long)
        ids[:rows, :length], mask[:rows, :length] = input_ids, attention_mask
        return FakeEncoder()(ids, mask).last_hidden_state[:rows]


def _adapter_with(runner: _Runner | None, **kwargs: Any) -> GLiNER2DecideAdapter:
    adapter, _ = make_adapter(**kwargs)
    adapter._graphs = runner
    return adapter


class TestAdapter:
    def test_a_call_that_fits_a_graph_scores_as_it_does_eagerly(self) -> None:
        runner = _PaddingRunner(window=256)
        adapter = _adapter_with(runner)
        rows = [[5] * n for n in (3, 40, 7, 61, 1)]
        positions = [0, 1, 2]

        out = adapter._score(rows, positions)

        np.testing.assert_allclose(
            out, np.array([expected_logits(row, positions) for row in rows]), rtol=1e-5, atol=1e-9
        )
        assert runner.replayed == [(6, 64)]  # one forward: five rows, padded to six of 64 tokens

    def test_a_call_that_fits_no_graph_runs_as_it_does_without_graphs(self) -> None:
        runner = _PaddingRunner(window=256)
        adapter = _adapter_with(runner, inference_batch_tokens=256)
        rows = [[5] * n for n in (3, 40, 7, 250, 1, 9, 9, 9, 9)]  # nine rows of 256 tokens pass the token bound
        positions = [0, 1, 2]

        out = adapter._score(rows, positions)

        np.testing.assert_allclose(
            out, np.array([expected_logits(row, positions) for row in rows]), rtol=1e-5, atol=1e-9
        )
        assert runner.stats.forwards == 0

    def test_a_graph_that_cannot_be_recorded_yet_runs_its_call_eagerly(self) -> None:
        runner = _PaddingRunner(window=256)
        runner.headroom = False
        adapter = _adapter_with(runner)
        rows = [[5] * n for n in (3, 40)]

        out = adapter._score(rows, [0, 1])

        np.testing.assert_allclose(out, np.array([expected_logits(row, [0, 1]) for row in rows]), rtol=1e-5, atol=1e-9)
        assert runner.stats.eager == {"no_headroom": 1}

    def test_rows_read_their_own_label_markers(self) -> None:
        runner = _PaddingRunner(window=256)
        adapter = _adapter_with(runner)
        rows = [[5] * 12, [7] * 30, [9] * 20]
        positions = [[0, 3], [1, 2, 4, 8], [5]]

        out = adapter._score_rows(rows, positions)

        assert [len(logits) for logits in out] == [2, 4, 1]
        for logits, row, markers in zip(out, rows, positions, strict=True):
            np.testing.assert_allclose(logits, expected_logits(row, markers), rtol=1e-5, atol=1e-9)
        assert runner.replayed == [(3, 32)]

    def test_rows_read_their_own_label_markers_without_graphs_too(self) -> None:
        adapter = _adapter_with(None, window=64, inference_batch_tokens=64)
        rows = [[5] * 12, [7] * 64, [9] * 20, [3] * 40]  # 64 tokens a forward: three forwards
        positions = [[0, 3], [1, 2, 4, 8], [5], [6, 7]]

        out = adapter._score_rows(rows, positions)

        for logits, row, markers in zip(out, rows, positions, strict=True):
            np.testing.assert_allclose(logits, expected_logits(row, markers), rtol=1e-5, atol=1e-9)

    def test_an_out_of_memory_error_drops_the_graphs(self) -> None:
        runner = _PaddingRunner(window=256)
        adapter = _adapter_with(runner)
        adapter._score([[5] * 10], [0])
        assert runner.graph_count == 1
        runner.replay_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            adapter._score([[5] * 10], [0])

        assert runner.graph_count == 0

    def test_other_errors_keep_the_graphs(self) -> None:
        runner = _PaddingRunner(window=256)
        adapter = _adapter_with(runner)
        adapter._score([[5] * 10], [0])
        runner.replay_error = RuntimeError("device-side assert")

        with pytest.raises(RuntimeError, match="device-side assert"):
            adapter._score([[5] * 10], [0])

        assert runner.graph_count == 1

    def test_unloading_drops_the_runner(self) -> None:
        assert "_graphs" in GLiNER2DecideAdapter.spec.unload_fields


class TestOperatorSetting:
    """Graphs are chosen when the model loads."""

    @pytest.mark.parametrize(("text", "mode"), [("off", "off"), ('"off"', "off"), ("bucketed", "bucketed")])
    def test_profiles_may_write_off_unquoted(self, text: str, mode: str) -> None:
        loadtime = yaml.safe_load(f"cuda_graphs: {text}")  # YAML reads unquoted off as False

        assert GLiNER2DecideAdapter("tiny", **loadtime)._cuda_graphs == mode

    def test_graphs_are_off_unless_the_operator_turns_them_on(self) -> None:
        assert GLiNER2DecideAdapter("tiny")._cuda_graphs == "off"

    @pytest.mark.parametrize("value", ["on", "exact", "Bucketed", "", None, True, 0])
    def test_unknown_modes_fail_the_load(self, value: Any) -> None:
        with pytest.raises(ValueError, match="cuda_graphs must be 'off' or 'bucketed'"):
            GLiNER2DecideAdapter("tiny", cuda_graphs=value)

    def test_profiles_may_set_it_at_load(self) -> None:
        reject_unknown_loadtime_options(GLiNER2DecideAdapter, {"cuda_graphs": "bucketed"}, model_name="tiny")

    def test_off_builds_no_runner(self) -> None:
        adapter = GLiNER2DecideAdapter("tiny")

        assert adapter._graph_runner(SimpleNamespace(encoder=_encoder()), "cuda") is None

    def test_an_encoder_graphs_do_not_apply_to_runs_eagerly_and_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = GLiNER2DecideAdapter("fastino/GLiNER2.5-Decide-1B", cuda_graphs="bucketed")

        with caplog.at_level(logging.WARNING):
            runner = adapter._graph_runner(SimpleNamespace(encoder=_encoder(encoder_type="modernbert")), "cuda")

        assert runner is None
        assert "cuda_graphs=bucketed does not apply to fastino/GLiNER2.5-Decide-1B" in caplog.text
        assert "the modernbert encoder is not supported" in caplog.text

    def test_a_supported_encoder_gets_a_runner_for_the_models_window(self) -> None:
        adapter = GLiNER2DecideAdapter("tiny", cuda_graphs="bucketed", max_seq_length=512)
        adapter._tokenizer = SimpleNamespace(pad_token_id=3)

        runner = adapter._graph_runner(SimpleNamespace(encoder=_encoder()), "cuda:0")

        assert runner is not None
        assert runner.shape_count == 132
        assert runner._pad_token_id == 3


def test_a_graph_keeps_to_the_operators_bound_on_a_forward_pass() -> None:
    adapter = GLiNER2DecideAdapter("tiny", cuda_graphs="bucketed", max_seq_length=512, inference_batch_tokens=640)
    adapter._tokenizer = SimpleNamespace(pad_token_id=0)

    runner = adapter._graph_runner(SimpleNamespace(encoder=_encoder()), "cuda")

    assert runner is not None
    assert runner.key(8, 60) == (8, 64)  # 512 padded tokens
    assert runner.key(12, 60) is None  # 768 would pass the 640 the operator allows a forward
    assert runner.key(1, 512) == (1, 512)
    assert all(batch * length <= 640 for batch, length in runner._shapes)


def test_shipped_profiles_enable_graphs_only_where_measured() -> None:
    configs = load_model_configs(_MODELS_DIR)
    modes = {
        name.split(":")[0]: config.resolve_profile("default").loadtime.get("cuda_graphs", "off")
        for name, config in configs.items()
        if config.resolve_profile("default").adapter_path.endswith(":GLiNER2DecideAdapter")
    }

    assert len(modes) == 3
    assert {name for name, mode in modes.items() if mode != "off"} == _BUCKETED_BY_DEFAULT
    for name in _BUCKETED_BY_DEFAULT:
        assert modes[name] == "bucketed"
        adapter = GLiNER2DecideAdapter(**_build_adapter_kwargs(configs[name], "float16"))
        assert adapter._cuda_graphs == "bucketed"
