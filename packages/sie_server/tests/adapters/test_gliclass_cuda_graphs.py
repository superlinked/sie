"""GLiClass CUDA graph policy: which forwards replay a graph, which record one, which run eagerly.

These run without a GPU: recording and replay are replaced by fakes. Scores
from real graphs are checked against eager scores in test_gliclass_parity.py
(``gpu_hw``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from gliclass.model import GLiClassUniEncoder
from sie_server.adapters.gliclass.cuda_graphs import (
    CudaGraphRunner,
    bucket_width,
    segment_ids,
    unsupported_reason,
)


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
    """Records and replays with fakes: logits say which path produced them."""

    def __init__(self, model: Any = None, *, fail: bool = False, **kwargs: Any) -> None:
        super().__init__(model or _model(), pad_token_id=0, max_length=512, **kwargs)
        self.recorded: list[tuple[int, int, int]] = []
        self.replayed: list[tuple[int, int, int]] = []
        self.fail = fail

    def _record(self, key: Any, inputs: dict[str, torch.Tensor]) -> Any:
        if self.fail:
            raise RuntimeError("operation not permitted when stream is capturing")
        self.recorded.append(key)
        return SimpleNamespace(key=key, replays=0), torch.zeros(key[0], key[2])

    def _replay(self, entry: Any, inputs: dict[str, torch.Tensor], length: int) -> torch.Tensor:
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

    def test_shapes_past_four_windows_run_eagerly(self) -> None:
        runner = _Runner()

        assert runner.key(4, 512, 4, "exact") == (4, 512, 4)
        assert runner.key(5, 512, 4, "exact") is None
        assert runner.key(64, 32, 4, "bucketed") == (64, 32, 4)
        assert runner.key(65, 32, 4, "bucketed") is None
        assert runner.key(5, 380, 8, "bucketed") == (5, 384, 8)
        assert runner.key(5, 400, 8, "bucketed") is None  # 5 x 416 tokens

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

    def test_recording_is_rationed_so_many_shapes_cannot_thrash(self) -> None:
        runner = _Runner(max_graphs=1000)

        # Every call is a new shape: sixteen record at once, then one per eight calls.
        answers = [runner.run(_inputs(1, 100), classes, "bucketed") for classes in range(2, 82)]

        assert 16 + 80 // 8 - 1 <= len(runner.recorded) <= 16 + 80 // 8
        assert answers.count(None) == 80 - len(runner.recorded)  # the rest ran eagerly

    def test_recording_slows_while_graphs_leave_the_cache_unused(self) -> None:
        runner = _Runner(max_graphs=4)

        # Shapes that never repeat: evicted graphs were never replayed.
        for classes in range(2, 402):
            runner.run(_inputs(1, 100), classes, "bucketed")
        wasteful = len(runner.recorded)
        # Fewer than one recording per 8 forwards past the first burst.
        assert wasteful < 16 + 400 // 8
        assert runner._wasted > 0.5

        # Shapes that repeat are still recorded, one per 64 forwards, then replayed.
        runner.recorded.clear()
        for _ in range(100):
            for classes in (500, 501):
                runner.run(_inputs(1, 100), classes, "bucketed")
        assert set(runner.recorded) == {(1, 128, 500), (1, 128, 501)}
        assert runner.replayed[-2:] == [(1, 128, 500), (1, 128, 501)]

    def test_a_recording_failure_turns_graphs_off(self) -> None:
        runner = _Runner(fail=True)

        assert runner.run(_inputs(1, 100), 4, "bucketed") is None
        assert runner.disabled
        runner.fail = False
        assert runner.run(_inputs(1, 100), 4, "bucketed") is None
        assert runner.recorded == []

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


def test_an_eager_out_of_memory_drops_the_graphs() -> None:
    from sie_server.adapters.gliclass import GLiClassAdapter

    class _Pipe:
        def _resolve_max_num_classes(self, labels: list[str], same_labels: bool) -> int:
            return len(labels)

        def model(self, **_: Any) -> Any:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    runner = _Runner()
    runner.run(_inputs(1, 100), 2, "bucketed")
    assert runner.graph_count == 1
    runner._recording_credit = 0.0  # the next shape is not recorded and runs eagerly
    adapter = GLiClassAdapter("tiny", max_seq_length=512)
    adapter._graphs = runner

    with pytest.raises(torch.cuda.OutOfMemoryError):
        adapter._forward(_Pipe(), _inputs(1, 300), ["a", "b"], same_labels=True, graphs="bucketed")

    assert runner.graph_count == 0
    assert runner.recorded == [(1, 128, 2)]
    assert not runner.disabled
