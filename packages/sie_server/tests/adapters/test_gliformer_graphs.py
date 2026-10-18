"""GLiFormer encoder-graph admission, storage ownership and release controls.

CPU policy tests replace capture/replay only. Tiny real encoders verify the
relative-position seam. CUDA tests exercise real capture when a GPU is present;
they are not a pretrained performance qualification.
"""

from __future__ import annotations

import gc
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from sie_server.adapters._cuda_graphs import RECORDING_LOCK
from sie_server.adapters.gliformer import adapter as adapter_module
from sie_server.adapters.gliformer import cuda_graphs as graphs
from sie_server.adapters.gliformer.adapter import GLiFormerAdapter, _import_gliformer
from sie_server.core.loader import reject_unknown_loadtime_options
from sie_server.types.inputs import Item
from transformers import BertTokenizerFast


class _Layers:
    def get_rel_pos(self, hidden: torch.Tensor, query_states: Any = None, relative_pos: Any = None) -> Any:
        return (hidden.shape[1], query_states) if relative_pos is None else relative_pos


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)
        self.model = SimpleNamespace(
            encoder=_Layers(),
            config=SimpleNamespace(hidden_size=4),
            training=False,
            get_input_embeddings=lambda: self.embedding,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.embedding(input_ids) * attention_mask[..., None]


class _PolicyRunner(graphs.EncoderGraphRunner):
    def __init__(self, **kwargs: Any) -> None:
        self.now = 0.0
        self.capture_error: Exception | None = None
        self.replay_error: Exception | None = None
        self.captures: list[tuple[int, int]] = []
        self.replays = 0
        super().__init__(
            _Transformer().eval(),
            mode=kwargs.pop("mode", "bucketed"),
            window=256,
            pad_token_id=0,
            clock=lambda: self.now,
            **kwargs,
        )

    def _capture(self, key: graphs.Key, device: torch.device) -> graphs._Graph:
        if self.capture_error is not None:
            raise self.capture_error
        self.captures.append(key)
        ids = torch.zeros(key, dtype=torch.long)
        self._relative_pos[key[1]] = torch.zeros(key[1], key[1], dtype=torch.long)
        return graphs._Graph(None, ids, ids.clone(), torch.zeros((*key, 4)), 100)

    def _replay(self, entry: graphs._Graph, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.replay_error is not None:
            raise self.replay_error
        self.replays += 1
        return input_ids[..., None].expand(-1, -1, 4).float().clone()

    def offer(self, rows: int = 1, length: int = 8) -> torch.Tensor | None:
        key = self.key(rows, length)
        assert key is not None
        ids = torch.ones(rows, length, dtype=torch.long)
        return self.run(ids, ids, key)


@pytest.fixture
def room(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state = {"headroom": True, "budget": 2**40}
    monkeypatch.setattr(graphs, "has_headroom", lambda _: state["headroom"])
    monkeypatch.setattr(graphs, "memory_budget", lambda _: state["budget"])
    return state


@pytest.mark.parametrize("mode", ["exact", "bucketed"])
def test_shape_policy_is_bounded_and_preserves_batch_rows(mode: str) -> None:
    for rows in (1, 2, 3, 8, 31, 64, 1024):
        for length in (1, 15, 31, 32, 33, 100, 255, 256, 257):
            key = graphs.graph_shape(rows, length, mode=mode, window=256, max_tokens=1024)
            if key is not None:
                assert key[0] == rows
                assert length <= key[1] <= 256
                assert key[0] * key[1] <= 1024
                assert key[1] == (length if mode == "exact" else -(-length // 32) * 32)
    for shape in ((0, 4), (1, 0), (1, 257), (9, 256)):
        assert graphs.graph_shape(*shape, mode=mode, window=256, max_tokens=1024) is None


def test_second_sighting_records_and_cache_replays(room: dict[str, Any]) -> None:
    runner = _PolicyRunner(mode="exact")
    assert runner.offer() is None
    assert runner.offer() is not None
    assert runner.offer() is not None
    assert runner.captures == [(1, 8)]
    assert runner.stats.forwards == 3
    assert (runner.stats.recorded, runner.stats.replayed, runner.stats.eager) == (1, 1, {"first_shape": 1})
    runner.close()


def test_no_headroom_global_recording_lock_and_credit_limits(room: dict[str, Any]) -> None:
    runner = _PolicyRunner()
    room["headroom"] = False
    assert runner.offer() is None
    room["headroom"] = True
    with RECORDING_LOCK:
        assert runner.offer() is None
    runner._credit = 0
    assert runner.offer() is None
    runner.now = 2
    assert runner.offer() is not None
    assert set(runner.stats.eager) == {"no_headroom", "busy", "recording_paused"}
    runner.close()


def test_over_budget_capture_is_not_retained(room: dict[str, Any]) -> None:
    runner = _PolicyRunner()
    room["budget"] = 1
    assert runner.offer() is None
    assert runner.held_bytes == runner.graph_count == 0
    assert runner.replays == 0
    assert runner.offer() is None
    assert len(runner.captures) == 1
    runner.clear()
    room["budget"] = 2**40
    assert runner.offer() is not None
    runner.close()


@pytest.mark.parametrize(("free_after", "expected"), [(20000, 12288), (0, 15000)])
def test_private_pool_accounting_survives_unrelated_driver_frees(
    monkeypatch: pytest.MonkeyPatch, free_after: int, expected: int
) -> None:
    pool = (42, 0)
    snapshot = Mock(
        return_value=[
            {"segment_pool_id": pool, "device": 0, "total_size": 4096, "address": 4096},
            {"segment_pool_id": pool, "device": 0, "total_size": 8192, "address": 8192},
        ]
    )
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    monkeypatch.setattr(graphs, "free_memory", lambda _: free_after)
    assert graphs._capture_device_bytes(SimpleNamespace(pool=lambda: pool), torch.device("cuda:0"), 15000) == expected
    snapshot.assert_called_once_with(pool)


@pytest.mark.parametrize("invalid", ["missing", "error", "wrong_pool", "wrong_device", "negative", "duplicate"])
def test_failed_pool_accounting_cannot_publish_a_graph(
    room: dict[str, Any], monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    pool = (42, 0)
    segment = {"segment_pool_id": pool, "device": 0, "total_size": 4096, "address": 4096}
    segments = [segment]
    if invalid == "wrong_pool":
        segment["segment_pool_id"] = (43, 0)
    elif invalid == "wrong_device":
        segment["device"] = 1
    elif invalid == "negative":
        segment["total_size"] = -1
    elif invalid == "duplicate":
        segments.append(segment.copy())
    elif invalid == "missing":
        segments.clear()
    snapshot = (
        Mock(side_effect=RuntimeError("snapshot unavailable")) if invalid == "error" else Mock(return_value=segments)
    )
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    runner = _PolicyRunner()
    original_capture = runner._capture

    def accounted_capture(key: graphs.Key, device: torch.device) -> graphs._Graph:
        entry = original_capture(key, device)
        entry.device_bytes = graphs._capture_device_bytes(SimpleNamespace(pool=lambda: pool), torch.device("cuda:0"), 0)
        return entry

    monkeypatch.setattr(runner, "_capture", accounted_capture)
    assert runner.offer() is None
    assert runner.graph_count == runner.held_bytes == runner.replays == 0
    assert runner.stats.recording_failures == 1
    assert not RECORDING_LOCK.locked()
    runner.close()


def test_private_pool_total_rejects_capture_even_when_driver_memory_increases(
    room: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = (42, 0)
    monkeypatch.setattr(
        torch.cuda,
        "memory_snapshot",
        lambda _: [{"segment_pool_id": pool, "device": 0, "total_size": 8192, "address": 4096}],
    )
    monkeypatch.setattr(graphs, "free_memory", lambda _: 20000)
    runner = _PolicyRunner()
    original_capture = runner._capture

    def accounted_capture(key: graphs.Key, device: torch.device) -> graphs._Graph:
        entry = original_capture(key, device)
        entry.device_bytes = graphs._capture_device_bytes(SimpleNamespace(pool=lambda: pool), torch.device("cuda:0"), 0)
        return entry

    monkeypatch.setattr(runner, "_capture", accounted_capture)
    room["budget"] = 4096
    assert runner.offer() is None
    assert runner.graph_count == runner.held_bytes == runner.replays == 0
    assert runner.stats.eager == {"budget_full": 1}
    runner.close()


@pytest.mark.parametrize("oom", [False, True])
def test_capture_failure_restores_eager_and_bounds_retries(room: dict[str, Any], oom: bool) -> None:
    runner = _PolicyRunner()
    runner.capture_error = torch.OutOfMemoryError("allocation failed") if oom else RuntimeError("not graphable")
    for length in (1, 33, 65, 97):
        assert runner.offer(length=length) is None
    assert runner.held_bytes == 0
    if oom:
        assert runner.stats.drops == 1
        assert not runner._failed
        runner.now = 61
        runner.capture_error = None
        assert runner.offer() is not None
    else:
        assert len(runner._failed) == graphs._MAX_FAILURES
        assert runner._disabled
    assert not RECORDING_LOCK.locked()
    runner.close()


def test_existing_replay_oom_releases_cache_and_propagates(room: dict[str, Any]) -> None:
    runner = _PolicyRunner()
    assert runner.offer() is not None
    runner.replay_error = torch.OutOfMemoryError("allocation failed")
    with pytest.raises(torch.OutOfMemoryError):
        runner.offer()
    assert runner.held_bytes == 0
    assert runner._pause_until == 60
    runner.close()


def test_shape_histories_and_graphs_are_bounded(room: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graphs, "_MAX_SIGHTINGS", 3)
    monkeypatch.setattr(graphs, "_MAX_GRAPHS", 2)
    runner = _PolicyRunner(mode="exact")
    for length in range(1, 6):
        assert runner.offer(length=length) is None
        assert runner.offer(length=length) is not None
    assert len(runner._seen) == 3
    assert runner.graph_count == 2
    assert set(runner._relative_pos) == {4, 5}
    runner.close()


@pytest.mark.parametrize("instance_override", [False, True])
def test_close_restores_exact_attributes_and_releases_model(instance_override: bool) -> None:
    transformer = _Transformer().eval()
    layers = transformer.model.encoder
    if instance_override:
        transformer.forward = transformer.forward
        layers.get_rel_pos = layers.get_rel_pos
    before_forward = transformer.__dict__.get("forward")
    before_rel = layers.__dict__.get("get_rel_pos")
    forward = transformer.forward
    rel = layers.get_rel_pos
    runner = graphs.EncoderGraphRunner(transformer, mode="exact", window=64, pad_token_id=0)
    with pytest.raises(RuntimeError, match="test failure"), runner._recording(torch.ones(1)):
        raise RuntimeError("test failure")
    assert not runner._local.recording
    runner.close()
    runner.close()
    assert transformer.__dict__.get("forward") is before_forward
    assert layers.__dict__.get("get_rel_pos") is before_rel
    assert transformer.forward.__func__ is forward.__func__
    assert transformer.forward.__self__ is transformer
    assert layers.get_rel_pos.__func__ is rel.__func__
    assert layers.get_rel_pos.__self__ is layers
    ref = weakref.ref(transformer)
    del runner, transformer, forward, before_forward
    gc.collect()
    assert ref() is None


def test_relative_position_override_is_local_to_recording_thread() -> None:
    runner = _PolicyRunner()
    observed: list[Any] = []
    hidden = torch.zeros(1, 7, 4)
    with runner._recording("static-table"):
        assert runner._layers.get_rel_pos(hidden) == "static-table"
        thread = threading.Thread(target=lambda: observed.append(runner._layers.get_rel_pos(hidden)))
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert observed == [(7, None)]
    assert runner._layers.get_rel_pos(hidden) == (7, None)
    runner.close()


def test_replay_copies_and_slices_outputs_on_the_owned_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real replay method with CPU storage and an explicit fake CUDA queue."""
    runner = _PolicyRunner()
    calls: list[Any] = []
    stream = SimpleNamespace(wait_stream=lambda other: calls.append(("wait", other)))
    runner._stream = stream
    host_stream = SimpleNamespace(wait_stream=lambda other: calls.append(("return_wait", other)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: host_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda selected: nullcontext(calls.append(("stream", selected))))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, other: calls.append(("record", other)))
    ids = torch.zeros(2, 32, dtype=torch.long)
    mask = torch.zeros_like(ids)
    hidden = torch.empty(2, 32, 4)
    graph = SimpleNamespace(replay=lambda: hidden.copy_((ids * mask)[..., None]))
    entry = graphs._Graph(graph, ids, mask, hidden, 0)
    first_ids = torch.arange(14).reshape(2, 7).long()
    first = graphs.EncoderGraphRunner._replay(runner, entry, first_ids, torch.ones_like(first_ids))
    second_ids = torch.full((2, 3), 42, dtype=torch.long)
    second = graphs.EncoderGraphRunner._replay(runner, entry, second_ids, torch.ones_like(second_ids))
    assert torch.equal(first, first_ids[..., None].expand(-1, -1, 4).float())
    assert tuple(first.shape) == (2, 7, 4)
    assert tuple(second.shape) == (2, 3, 4)
    assert first.untyped_storage().data_ptr() != entry.hidden.untyped_storage().data_ptr()
    assert torch.count_nonzero(entry.attention_mask[:, 3:]) == 0
    assert torch.count_nonzero(entry.input_ids[:, 3:]) == 0
    assert [call[0] for call in calls] == ["wait", "stream", "return_wait", "record"] * 2
    assert calls[0][1] is host_stream
    assert calls[2][1] is stream
    assert calls[3][1] is host_stream
    runner._stream = None
    runner.close()


def test_pressure_clear_waits_before_dropping_graph_storage(room: dict[str, Any]) -> None:
    runner = _PolicyRunner()
    assert runner.offer() is not None
    observed: list[int] = []
    runner._stream = SimpleNamespace(synchronize=lambda: observed.append(runner.graph_count))
    runner.clear()
    assert observed == [1]
    assert runner.held_bytes == 0
    runner._stream = None
    runner.close()


@pytest.mark.parametrize("mode", ["off", False, "exact", "bucketed"])
def test_operator_option_and_default_eager_path(mode: str | bool, monkeypatch: pytest.MonkeyPatch) -> None:
    reject_unknown_loadtime_options(GLiFormerAdapter, {"cuda_graphs": mode}, model_name="tiny")
    adapter = GLiFormerAdapter("tiny", cuda_graphs=mode)
    assert GLiFormerAdapter("tiny")._cuda_graphs == "off"
    if mode in ("off", False):
        monkeypatch.setattr(adapter_module, "unsupported_reason", Mock(side_effect=AssertionError("default changed")))
        assert adapter._graph_runner(None, None, "cuda") is None
    else:
        assert adapter._graph_runner(SimpleNamespace(model=None), None, "cpu") is None
    assert not adapter.has_releasable_memory()
    assert adapter.release_optional_memory() == 0


@pytest.mark.parametrize("mode", [True, None, 1, "on", "EXACT", []])
def test_invalid_modes_fail_before_loading(mode: Any) -> None:
    with pytest.raises(ValueError, match="cuda_graphs"):
        GLiFormerAdapter("tiny", cuda_graphs=mode)


def test_pressure_release_keeps_weights_and_unload_restores_forward(room: dict[str, Any]) -> None:
    runner = _PolicyRunner()
    adapter = GLiFormerAdapter("tiny", cuda_graphs="bucketed")
    transformer = runner._transformer
    adapter._model = transformer
    adapter._graphs = runner
    assert runner.offer() is not None
    held = runner.held_bytes
    assert adapter.has_releasable_memory()
    assert adapter.release_optional_memory() == held > 0
    assert adapter._model is transformer
    assert runner.graph_count == 0
    assert "forward" in transformer.__dict__
    assert adapter.release_optional_memory() == 0
    adapter.unload()
    assert adapter._model is None
    assert adapter._graphs is None
    assert "forward" not in transformer.__dict__


def _tiny_transformer(layout: bool, fuse: bool = False) -> Any:
    _import_gliformer()
    from gliformer.backbones.deberta_2d import LayoutDebertaConfig
    from gliformer.encoders.text import TextTransformer
    from transformers import DebertaV2Config

    config_class = LayoutDebertaConfig if layout else DebertaV2Config
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        relative_attention=True,
        max_relative_positions=64,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        max_position_embeddings=64,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        attn_kernel="eager",
        max_page_embeddings=8,
    )
    return TextTransformer(
        "unused-local-name",
        SimpleNamespace(
            encoder_config=config,
            backbone_type="deberta_2d" if layout else "auto",
            fuse_layers=fuse,
            _attn_implementation=None,
            max_page_embeddings=8,
        ),
        from_pretrained=False,
        local_files_only=True,
    ).eval()


@pytest.mark.parametrize("layout", [False, True], ids=["deberta", "layout-deberta"])
@pytest.mark.parametrize("fuse", [False, True])
def test_real_encoder_relative_position_seam_is_exact(layout: bool, fuse: bool) -> None:
    transformer = _tiny_transformer(layout, fuse)
    assert graphs.unsupported_reason(transformer, "cuda") is None
    ids = torch.arange(14).reshape(2, 7).long()
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    original = transformer.forward
    with torch.inference_mode():
        expected = original(input_ids=ids, attention_mask=mask)
        runner = graphs.EncoderGraphRunner(transformer, mode="exact", window=64, pad_token_id=0)
        table = runner._build_relative_pos(torch.empty((1, 7, 1)))
        with runner._recording(table):
            got = original(input_ids=ids, attention_mask=mask)
        assert torch.equal(got, expected)
        # A CPU call is deliberately eager and retains the real result.
        assert torch.equal(transformer(input_ids=ids, attention_mask=mask), expected)
        assert runner.graph_count == 0
        runner.close()


def test_training_custom_backbones_and_flash_are_not_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _tiny_transformer(True)
    transformer.train()
    assert "training" in str(graphs.unsupported_reason(transformer, "cuda"))
    transformer.eval()
    from gliformer.backbones import deberta_2d

    transformer.model.encoder.attn_kernel = "flash"
    monkeypatch.setattr(deberta_2d, "flash_kernels_available", lambda: True)
    assert "flash" in str(graphs.unsupported_reason(transformer, "cuda"))
    assert "unsupported" in str(graphs.unsupported_reason(_Transformer(), "cuda"))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local LayoutDeBERTa checkpoint with all four extraction heads and embeddings."""
    gliformer = _import_gliformer()
    root = tmp_path_factory.mktemp("gliformer-graphs")
    vocab = root / "vocab.txt"
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "Alice",
        "Bob",
        "works",
        "at",
        "Acme",
        "likes",
        "coffee",
        ".",
        "person",
        "organization",
        "review",
        "allow",
        "block",
        "works_for",
    ]
    vocab.write_text("\n".join(vocabulary))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    config = gliformer.GLiFormerConfig(
        model_name="unused-local-name",
        hidden_size=32,
        max_len=128,
        vocab_size=len(tokenizer),
        backbone_type="deberta_2d",
        model_variant="layout",
        use_layout=True,
        layout_image_tokens=False,
        max_page_embeddings=8,
        num_rnn_layers=1,
        encoder_config={
            "model_type": "layout-deberta",
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "relative_attention": True,
            "position_buckets": 16,
            "max_relative_positions": 64,
            "pos_att_type": ["p2c", "c2p"],
            "position_biased_input": False,
            "max_position_embeddings": 128,
            "hidden_dropout_prob": 0,
            "attention_probs_dropout_prob": 0,
            "attn_kernel": "eager",
        },
        ner_config={},
        classification_config={},
        joint_relex_config={},
        structuring_config={},
        embedding_config={"projection_dim": 32},
    )
    with torch.random.fork_rng():
        torch.manual_seed(374)
        model = gliformer.GLiFormerLayout(
            config,
            tokenizer=tokenizer,
            backbone_from_pretrained=False,
            local_files_only=True,
        ).eval()
        model.resize_embeddings()
    checkpoint = root / "checkpoint"
    model.save_pretrained(checkpoint)
    return checkpoint


class _EagerReplay(graphs.EncoderGraphRunner):
    """CPU-only replay emulation: real encoder and heads, no CUDA execution claim."""

    def _inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        assert not args
        assert set(kwargs) == {"input_ids", "attention_mask"}
        return kwargs["input_ids"], kwargs["attention_mask"]

    def _capture(self, key: graphs.Key, device: torch.device) -> graphs._Graph:
        ids = torch.zeros(key, dtype=torch.long)
        self._relative_pos[key[1]] = self._build_relative_pos(torch.empty((1, key[1], 1)))
        return graphs._Graph(None, ids, ids.clone(), torch.zeros((*key, self._hidden_size)), 0)

    def _replay(self, entry: graphs._Graph, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        ids = torch.nn.functional.pad(input_ids, (0, entry.input_ids.shape[1] - length))
        mask = torch.nn.functional.pad(attention_mask, (0, entry.input_ids.shape[1] - length))
        with self._recording(self._relative_pos[entry.input_ids.shape[1]]):
            return self._forward(input_ids=ids, attention_mask=mask)[:, :length].clone()


_TASKS: list[dict[str, Any]] = [
    {"labels": ["person", "organization"]},
    {"labels": ["allow", "block"], "options": {"classification_task": "review"}},
    {"labels": ["person", "organization"], "options": {"relation_labels": ["works_for"]}},
    {"output_schema": {"type": "object", "properties": {"person": {"type": "string"}}}},
    {"labels": ["person"], "options": {"label_groups": {"review": ["allow", "block"]}}},
    {
        "labels": ["person", "organization"],
        "options": {"relation_labels": ["works_for"]},
        "output_schema": {
            "type": "object",
            "properties": {"person": {"type": "string"}, "review": {"type": "string", "enum": ["allow", "block"]}},
        },
    },
]


def _assert_output_close(actual: Any, expected: Any, tolerance: float) -> None:
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=tolerance, rel=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_output_close(actual[key], expected[key], tolerance)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            _assert_output_close(first, second, tolerance)
    else:
        assert actual == expected


@pytest.mark.parametrize("mode", ["exact", "bucketed"])
def test_real_all_task_heads_and_usage_survive_encoder_only_padding(
    tiny_checkpoint: Path,
    mode: str,
    room: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual SIE load/inference/unload path on a generated checkpoint."""
    baseline = GLiFormerAdapter(tiny_checkpoint, max_seq_length=128, compute_precision="float32")
    candidate = GLiFormerAdapter(tiny_checkpoint, max_seq_length=128, compute_precision="float32", cuda_graphs=mode)
    monkeypatch.setattr(adapter_module, "unsupported_reason", lambda *_: None)
    monkeypatch.setattr(adapter_module, "EncoderGraphRunner", _EagerReplay)
    items = [Item(text="Alice works at Acme."), Item(text="Bob likes coffee.")]
    baseline.load("cpu")
    candidate.load("cpu")
    assert candidate._graphs is not None
    observed: list[dict[str, torch.Tensor]] = []

    def observe(_module: Any, _inputs: Any, output: Any) -> None:
        observed.append({key: value.clone() for key, value in output.items() if isinstance(value, torch.Tensor)})

    hook = candidate._model.model.register_forward_hook(observe)
    try:
        tolerance = 0 if mode == "exact" else 1e-5  # synthetic fp32 padding, not a pretrained/GPU bound
        for task in _TASKS:
            expected = baseline.extract(items, **task)
            candidate.extract(items, **task)  # exact mode's first sighting
            candidate.extract(items, **task)  # capture, then test an ordinary replay
            actual = candidate.extract(items, **task)
            for field in ("entities", "classifications", "relations", "data", "errors", "input_token_counts"):
                _assert_output_close(getattr(actual, field), getattr(expected, field), tolerance)
        eager_dense = baseline.encode(items, ["dense"]).dense
        candidate.encode(items, ["dense"])
        candidate.encode(items, ["dense"])
        dense = candidate.encode(items, ["dense"]).dense
        torch.testing.assert_close(torch.from_numpy(dense), torch.from_numpy(eager_dense), rtol=0, atol=tolerance)
        assert observed
        assert {"ner_logits", "cat_logits", "joint_rel_logits", "structuring_logits"} <= set().union(
            *(set(row) for row in observed)
        )
        for name in ("ner_logits", "cat_logits", "joint_rel_logits", "structuring_logits"):
            assert any(row[name].numel() > 0 for row in observed if name in row)
        assert candidate._graphs.stats.recorded > 0
        assert candidate._graphs.stats.replayed > 0
        transformer = candidate._model.model.token_rep_layer.bert_layer
        parameters = {name: id(value) for name, value in transformer.named_parameters()}
        assert candidate.release_optional_memory() > 0
        assert parameters == {name: id(value) for name, value in transformer.named_parameters()}
    finally:
        hook.remove()
        candidate.unload()
        baseline.unload()
    assert "forward" not in transformer.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA graph parity needs a GPU")
@pytest.mark.parametrize("layout", [False, True], ids=["deberta", "layout-deberta"])
@pytest.mark.parametrize("mode", ["exact", "bucketed"])
def test_actual_cuda_capture_matches_eager_padded_shape_and_owns_output(layout: bool, mode: str) -> None:
    transformer = _tiny_transformer(layout).to(device="cuda", dtype=torch.float16)
    original = transformer.forward
    ids = torch.arange(14, device="cuda").reshape(2, 7).long()
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    runner = graphs.EncoderGraphRunner(transformer, mode=mode, window=64, pad_token_id=0)
    try:
        with torch.inference_mode():
            key = runner.key(*ids.shape)
            assert key is not None
            padded_ids = torch.nn.functional.pad(ids, (0, key[1] - ids.shape[1]))
            padded_mask = torch.nn.functional.pad(mask, (0, key[1] - ids.shape[1]))
            expected = original(input_ids=padded_ids, attention_mask=padded_mask)[:, : ids.shape[1]]
            transformer(input_ids=ids, attention_mask=mask)
            got = transformer(input_ids=ids, attention_mask=mask)
            assert runner.stats.recorded == 1
            assert runner.graph_count == 1
            assert torch.equal(got, expected)
            kept = got.clone()
            transformer(input_ids=ids.flip(1), attention_mask=mask)
            assert torch.equal(got, kept)
            # Revisit shapes in reverse capture order. Separate private pools
            # make this safe without assuming a fixed request shape sequence.
            longer_ids = torch.arange(70, device="cuda").reshape(2, 35).remainder(64).long()
            longer_mask = torch.ones_like(longer_ids)
            longer_key = runner.key(*longer_ids.shape)
            assert longer_key is not None
            longer_expected = original(
                input_ids=torch.nn.functional.pad(longer_ids, (0, longer_key[1] - longer_ids.shape[1])),
                attention_mask=torch.nn.functional.pad(longer_mask, (0, longer_key[1] - longer_ids.shape[1])),
            )[:, : longer_ids.shape[1]]
            transformer(input_ids=longer_ids, attention_mask=longer_mask)
            assert torch.equal(transformer(input_ids=longer_ids, attention_mask=longer_mask), longer_expected)
            assert runner.stats.recorded == 2
            assert len({entry.graph.pool() for entry in runner._graphs.values()}) == 2
            assert torch.equal(transformer(input_ids=ids, attention_mask=mask), expected)
            assert torch.equal(transformer(input_ids=longer_ids, attention_mask=longer_mask), longer_expected)
            assert runner.held_bytes <= graphs.memory_budget(ids.device)
            allocated_before_clear = torch.cuda.memory_allocated(ids.device)
            runner.clear()
            assert runner.held_bytes == 0
            assert torch.cuda.memory_allocated(ids.device) < allocated_before_clear
    finally:
        runner.close()
