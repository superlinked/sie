"""GLiClass CUDA graph policy: which forwards replay a graph, which record one, which run eagerly.

These run without a GPU: recording and replay are replaced by fakes. Scores
from real graphs are checked against eager scores in test_gliclass_parity.py
(``gpu_hw``).
"""

from __future__ import annotations

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
    bucket_width,
    segment_ids,
    unsupported_reason,
)
from sie_server.core.loader import _build_adapter_kwargs, load_model_configs, reject_unknown_loadtime_options
from sie_server.types.inputs import InvalidInputError

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# The shipped profiles that load with graphs: DeBERTa-v3 models whose bucketed
# scores changed no top label against eager execution (see the server README).
_BUCKETED_BY_DEFAULT = {
    "knowledgator/gliclass-base-v1.0",
    "knowledgator/gliclass-large-v1.0",
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
) -> Any:
    config = SimpleNamespace(
        architecture_type=architecture,
        encoder_config=SimpleNamespace(model_type=encoder_type),
        pooling_strategy=pooling,
        use_lstm=False,
        use_segment_embeddings=segment_embeddings,
        example_token_index=9,
        text_token_index=8,
    )
    inner = SimpleNamespace(encoder_model=SimpleNamespace(encoder=_Encoder()), _create_segment_ids=lambda ids: "eager")
    return SimpleNamespace(config=config, model=inner)


class _Runner(CudaGraphRunner):
    """Records and replays with fakes: logits say which path produced them.

    Recording answers with zeros, a replay with ones, and a recording whose
    capture fails (``capture_error``) with the eager answer, twos. ``fail``
    makes the eager forward of a recording fail instead.

    Time stands still unless a test moves ``now``; the device always has
    memory to spare unless a test clears ``headroom`` or sets a ``budget``.
    """

    def __init__(self, model: Any = None, *, fail: bool | Exception = False, **kwargs: Any) -> None:
        self.now = 0.0
        super().__init__(model or _model(), pad_token_id=0, max_length=512, clock=lambda: self.now, **kwargs)
        self.recorded: list[tuple[int, int, int]] = []
        self.replayed: list[tuple[int, int, int]] = []
        self.fail = fail
        self.capture_error: Exception | None = None
        self.headroom = True
        self.budget = 2**40
        self.bytes_per_graph = 0
        self.replay_error: Exception | None = None

    def _has_headroom(self, device: torch.device) -> bool:
        return self.headroom

    def _memory_budget(self, device: torch.device) -> int:
        return self.budget

    def _record(self, key: Any, inputs: dict[str, torch.Tensor]) -> Any:
        if isinstance(self.fail, Exception):
            raise self.fail
        if self.fail:
            raise RuntimeError("the eager forward failed")
        if self.capture_error is not None:
            raise cuda_graphs_module._RecordingError(self.capture_error, torch.full((key[0], key[2]), 2.0))
        self.recorded.append(key)
        self._relative_pos.setdefault(key[1], torch.zeros(1))
        entry = SimpleNamespace(key=key, device_bytes=self.bytes_per_graph, inputs={}, output=torch.zeros(0))
        return entry, torch.zeros(key[0], key[2])

    def _replay(self, entry: Any, inputs: dict[str, torch.Tensor], length: int) -> torch.Tensor:
        if self.replay_error is not None:
            raise self.replay_error
        self.replayed.append(entry.key)
        return torch.ones(entry.key[0], entry.key[2])


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

    @pytest.mark.parametrize(("length", "bucketed"), [(1, 32), (32, 32), (33, 64), (500, 512), (512, 512)])
    def test_bucketed_lengths_round_up_to_the_bucket(self, length: int, bucketed: int) -> None:
        runner = _Runner()

        assert runner.key(1, length, 8, "bucketed") == (1, bucketed, 8)
        assert runner.key(1, length, 8, "exact") == (1, length, 8)

    def test_shapes_past_2048_tokens_run_eagerly(self) -> None:
        runner = _Runner()

        assert runner.key(4, 512, 4, "exact") == (4, 512, 4)
        assert runner.key(5, 512, 4, "exact") is None
        assert runner.key(64, 32, 4, "bucketed") == (64, 32, 4)
        assert runner.key(65, 32, 4, "bucketed") is None
        assert runner.key(5, 380, 8, "bucketed") == (5, 384, 8)
        assert runner.key(5, 400, 8, "bucketed") is None  # 5 x 416 tokens

    def test_the_token_bound_does_not_grow_with_the_window(self) -> None:
        runner = CudaGraphRunner(_model(), pad_token_id=0, max_length=1024)

        assert runner.key(2, 1024, 4, "exact") == (2, 1024, 4)
        assert runner.key(3, 1024, 4, "exact") is None
        assert runner.key(3, 600, 4, "bucketed") == (3, 640, 4)  # 64-token buckets

    @pytest.mark.parametrize("pooling", ["avg", "last", "max"])
    def test_poolings_that_read_padding_keep_exact_lengths(self, pooling: str) -> None:
        runner = _Runner(_model(pooling=pooling))

        assert runner.key(1, 100, 4, "bucketed") == (1, 100, 4)


class TestRecordingPolicy:
    def test_off_never_records(self) -> None:
        runner = _Runner()

        assert runner.run(_inputs(1, 100), 4, "off") is None
        assert runner.recorded == []

    def test_exact_mode_records_a_shape_the_second_time_it_is_seen(self) -> None:
        runner = _Runner()

        first = runner.run(_inputs(1, 100), 4, "exact")
        second = runner.run(_inputs(1, 100), 4, "exact")
        third = runner.run(_inputs(1, 100), 4, "exact")

        assert first is None  # eager
        assert second is not None
        assert torch.equal(second, torch.zeros(1, 4))  # the recording forward answers
        assert third is not None
        assert torch.equal(third, torch.ones(1, 4))  # a replay
        assert runner.recorded == [(1, 100, 4)]
        assert runner.replayed == [(1, 100, 4)]

    def test_bucketed_mode_records_once_per_bucket(self) -> None:
        runner = _Runner()

        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.run(_inputs(1, 110), 4, "bucketed")
        runner.run(_inputs(1, 128), 4, "bucketed")
        runner.run(_inputs(1, 129), 4, "bucketed")

        assert runner.recorded == [(1, 128, 4), (1, 160, 4)]
        assert runner.replayed == [(1, 128, 4), (1, 128, 4)]

    def test_batch_size_and_class_slots_are_part_of_the_shape(self) -> None:
        runner = _Runner()

        for batch, classes in ((1, 4), (2, 4), (1, 8)):
            runner.run(_inputs(batch, 100), classes, "bucketed")

        assert runner.recorded == [(1, 128, 4), (2, 128, 4), (1, 128, 8)]

    def test_the_least_recently_used_graph_is_dropped_past_the_bound(self) -> None:
        runner = _Runner(max_graphs=2)

        runner.run(_inputs(1, 32), 4, "bucketed")
        runner.run(_inputs(1, 64), 4, "bucketed")
        runner.run(_inputs(1, 32), 4, "bucketed")  # 32 is now the most recent
        runner.run(_inputs(1, 96), 4, "bucketed")  # drops 64
        runner.run(_inputs(1, 64), 4, "bucketed")  # recorded again

        assert runner.graph_count == 2
        assert runner.recorded == [(1, 32, 4), (1, 64, 4), (1, 96, 4), (1, 64, 4)]
        # A length's relative-position table lives as long as a graph of that length.
        assert set(runner._relative_pos) == {64, 96}

    def test_recording_is_rationed_in_time(self) -> None:
        runner = _Runner(max_graphs=1000)
        shapes = iter(range(2, 10_000))

        # Every call is a new shape: sixteen record at once, then none while time stands still.
        answers = [runner.run(_inputs(1, 100), next(shapes), "bucketed") for _ in range(80)]
        assert len(runner.recorded) == 16
        assert answers.count(None) == 64  # the rest ran eagerly

        runner.now += 2.0  # one more recording every 2 seconds
        for _ in range(10):
            runner.run(_inputs(1, 100), next(shapes), "bucketed")
        assert len(runner.recorded) == 17

        runner.now += 3600.0  # an idle hour refills only the burst
        for _ in range(40):
            runner.run(_inputs(1, 100), next(shapes), "bucketed")
        assert len(runner.recorded) == 33

    def test_repeating_each_shape_does_not_buy_recordings(self) -> None:
        runner = _Runner()

        # Each new shape sent twice, one forward per 10 ms for 80 seconds.
        for classes in range(2, 4002):
            for _ in range(2):
                runner.run(_inputs(1, 100), classes, "bucketed")
                runner.now += 0.01

        # The burst, then one per 2 seconds of the 80.
        assert 16 + 80 / 2 - 1 <= len(runner.recorded) <= 16 + 80 / 2

    def test_one_recording_at_a_time_in_the_process(self) -> None:
        runner = _Runner()

        assert cuda_graphs_module._RECORDING_LOCK.acquire(blocking=False)  # another model is recording
        try:
            assert runner.run(_inputs(1, 100), 4, "bucketed") is None
        finally:
            cuda_graphs_module._RECORDING_LOCK.release()
        assert runner.recorded == []
        assert runner.run(_inputs(1, 100), 4, "bucketed") is not None
        assert runner.recorded == [(1, 128, 4)]
        assert not cuda_graphs_module._RECORDING_LOCK.locked()

    def test_graphs_are_dropped_when_their_memory_passes_the_budget(self) -> None:
        runner = _Runner()
        runner.budget, runner.bytes_per_graph = 250, 100

        runner.run(_inputs(1, 32), 4, "bucketed")
        runner.run(_inputs(1, 64), 4, "bucketed")
        assert runner.graph_count == 2
        runner.run(_inputs(1, 96), 4, "bucketed")  # 300 bytes: over the budget

        assert runner.graph_count == 0
        assert runner._relative_pos == {}
        assert not runner.disabled
        runner.run(_inputs(1, 32), 4, "bucketed")  # records again from nothing
        assert runner.recorded[-1] == (1, 32, 4)
        assert runner.graph_count == 1

    def test_the_budget_counts_the_tensors_graphs_keep(self) -> None:
        runner = _Runner()
        runner.budget = 999
        runner._relative_pos[32] = torch.zeros(1000, dtype=torch.uint8)  # a 1,000-byte table

        runner.run(_inputs(1, 32), 4, "bucketed")  # nothing recorded counts, but the table does

        assert runner.graph_count == 0

    def test_evicting_graphs_does_not_give_their_memory_back(self) -> None:
        # Evicted graphs leave their share of the shared pool behind, so their
        # memory still counts until the graphs are dropped.
        runner = _Runner(max_graphs=1)
        runner.budget, runner.bytes_per_graph = 250, 100

        runner.run(_inputs(1, 32), 4, "bucketed")
        runner.run(_inputs(1, 64), 4, "bucketed")
        assert runner.graph_count == 1
        runner.run(_inputs(1, 96), 4, "bucketed")

        assert runner.graph_count == 0

    def test_no_recording_while_device_memory_is_short(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.headroom = False

        assert runner.run(_inputs(1, 200), 4, "bucketed") is None  # would record: runs eagerly
        assert runner.run(_inputs(1, 100), 4, "bucketed") is not None  # replays still run
        assert runner.recorded == [(1, 128, 4)]
        assert runner.replayed == [(1, 128, 4)]

    def test_a_recording_failure_turns_graphs_off_and_keeps_the_eager_answer(self) -> None:
        runner = _Runner()
        runner.capture_error = RuntimeError("operation not permitted when stream is capturing")

        answer = runner.run(_inputs(1, 100), 4, "bucketed")

        assert answer is not None
        assert torch.equal(answer, torch.full((1, 4), 2.0))  # the eager answer
        assert runner.disabled
        assert not cuda_graphs_module._RECORDING_LOCK.locked()
        runner.capture_error = None
        assert runner.run(_inputs(1, 100), 4, "bucketed") is None
        assert runner.recorded == []

    def test_an_eager_forward_that_fails_while_recording_turns_graphs_off(self) -> None:
        runner = _Runner(fail=True)

        assert runner.run(_inputs(1, 100), 4, "bucketed") is None  # the adapter runs it eagerly
        assert runner.disabled
        assert not cuda_graphs_module._RECORDING_LOCK.locked()

    def test_running_out_of_memory_while_capturing_keeps_the_eager_answer_and_pauses(self) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.capture_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

        answer = runner.run(_inputs(1, 200), 4, "bucketed")

        assert answer is not None
        assert torch.equal(answer, torch.full((1, 4), 2.0))  # the eager answer, not an error
        assert runner.graph_count == 0
        assert not runner.disabled
        assert not cuda_graphs_module._RECORDING_LOCK.locked()
        # Recording pauses for a minute, then resumes.
        runner.capture_error = None
        runner.now += 59.0
        assert runner.run(_inputs(1, 300), 4, "bucketed") is None
        runner.now += 2.0
        assert runner.run(_inputs(1, 300), 4, "bucketed") is not None
        assert runner.recorded[-1] == (1, 320, 4)

    @pytest.mark.parametrize(
        "error",
        [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("CUBLAS_STATUS_ALLOC_FAILED: failed to allocate workspace"),
        ],
    )
    def test_an_eager_forward_out_of_memory_while_recording_drops_the_graphs_and_raises(self, error: Exception) -> None:
        runner = _Runner()
        runner.run(_inputs(1, 100), 4, "bucketed")
        runner.fail = error

        with pytest.raises(type(error)):
            runner.run(_inputs(1, 200), 4, "bucketed")

        assert runner.graph_count == 0
        assert not runner.disabled  # memory pressure is not a reason to stop recording
        assert not cuda_graphs_module._RECORDING_LOCK.locked()

    def test_unexpected_inputs_run_eagerly(self) -> None:
        runner = _Runner()
        inputs = {**_inputs(1, 100), "class_input_ids": torch.ones(1, 4, dtype=torch.long)}

        assert runner.run(inputs, 4, "bucketed") is None
        assert runner.recorded == []


class TestRecordingHooks:
    def test_hooks_leave_eager_forwards_alone(self) -> None:
        model = _model(segment_embeddings=True)
        _Runner(model)
        encoder = model.model.encoder_model.encoder
        hidden = torch.zeros(1, 7, 4)

        assert encoder.get_rel_pos(hidden) == ("built", 7)
        assert model.model._create_segment_ids(torch.ones(1, 3, dtype=torch.long)) == "eager"

    def test_while_recording_the_encoder_gets_the_precomputed_table(self) -> None:
        model = _model(segment_embeddings=True)
        runner = _Runner(model)
        encoder = model.model.encoder_model.encoder
        table = torch.arange(4)
        runner._recording, runner._recording_relative_pos = True, table

        assert encoder.get_rel_pos(torch.zeros(1, 7, 4)) is table
        assert encoder.get_rel_pos(torch.zeros(1, 7, 4), None, "given") == ("built", 7)
        ids = torch.tensor([[0, 8, 5, 5, 2]])
        assert torch.equal(model.model._create_segment_ids(ids), torch.tensor([[0, 1, 1, 1, 1]]))


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
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or torch.cuda.OutOfMemoryError("CUDA out of memory")

    def _resolve_max_num_classes(self, labels: list[str], same_labels: bool) -> int:
        return len(labels)

    def model(self, **_: Any) -> Any:
        raise self.error


def _adapter_with(runner: _Runner) -> GLiClassAdapter:
    adapter = GLiClassAdapter("tiny", max_seq_length=512, cuda_graphs="bucketed")
    adapter._graphs = runner
    return adapter


def test_an_eager_out_of_memory_drops_the_graphs() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    assert runner.graph_count == 1
    runner._recording_credit = 0.0  # the next shape is not recorded and runs eagerly

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _adapter_with(runner)._forward(_Pipe(), _inputs(1, 300), ["a", "b"], same_labels=True, graphs="bucketed")

    assert runner.graph_count == 0
    assert runner.recorded == [(1, 128, 2)]
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


def test_a_capture_out_of_memory_answers_eagerly_without_oom_recovery() -> None:
    runner = _Runner()
    runner.capture_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

    # The fake pipe's own forward would raise: the answer must come from the recording.
    logits = _adapter_with(runner)._forward(_Pipe(), _inputs(1, 100), ["a", "b"], same_labels=True, graphs="bucketed")

    assert torch.equal(logits, torch.full((1, 2), 2.0))
    assert runner.graph_count == 0
    assert not runner.disabled


def test_an_out_of_memory_error_in_a_replay_drops_the_graphs() -> None:
    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    runner.replay_error = torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _adapter_with(runner)._forward(_Pipe(), _inputs(1, 100), ["a", "b"], same_labels=True, graphs="bucketed")

    assert runner.graph_count == 0


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

        with pytest.raises(torch.cuda.OutOfMemoryError):  # the fake pipe's eager forward
            _adapter_with(runner)._forward(_Pipe(), _inputs(1, 100), ["a", "b"], same_labels=True, graphs="off")

        assert runner.replayed == []


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
