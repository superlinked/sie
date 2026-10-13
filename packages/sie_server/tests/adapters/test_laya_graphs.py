"""The Laya adapter replaying its encoder as CUDA graphs.

The CPU tests use the stand-ins of ``test_modernbert_flash_graphs``: a recorded
graph is replaced by one whose replay runs the recorded encoder again on the
graph's static inputs, and a reference attention stands in for flash-attn's
kernel. That attention writes NaN to every row outside a sequence, as the real
kernel leaves them unwritten, so a padding token that reached a real row would
fail every comparison. They check what the adapter records (its packed encoder
on a float32 residual stream), what it makes of a replay (the decision head on
the real rows) and when it replays. The ``gpu_hw`` test records and replays
real graphs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from sie_server.adapters._modernbert_flash_graphs import parse_graph_mode, token_buckets
from sie_server.adapters.laya.adapter import LayaAdapter, graph_token_bound
from sie_server.adapters.laya.model import LayaModel
from sie_server.adapters.laya.questions import QTYPE_NAMES
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options

from .test_laya_model import (
    MARKERS,
    PAD_ID,
    QTYPES,
    ROWS,
    _encoder_config,
    _load_like_adapter,
    reference_checkpoint,
)
from .test_modernbert_flash_graphs import (
    _Runner,
    reference_flash,
)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_LAYA_ADAPTER = "sie_server.adapters.laya.adapter:LayaAdapter"
_SHIPPED = {
    "convaiinnovations/laya",
    "convaiinnovations/laya-multilingual",
    "convaiinnovations/laya-typed-decisions",
}
_WINDOW = 64
# The adapter's own tolerance (``test_laya_model``): float32 logits against the reference forward.
_TOLERANCE = 1e-5


def _prefixes() -> list[Any]:
    """One row prefix per ``ROWS`` entry: its option markers and question type."""
    return [
        SimpleNamespace(markers=markers, question=SimpleNamespace(qtype=QTYPE_NAMES[qtype]))
        for markers, qtype in zip(MARKERS, QTYPES, strict=True)
    ]


def _flash_model(checkpoint: Path, *, fused_rope: bool = False) -> LayaModel:
    """The served model on its packed path, in float32 on the CPU."""
    model, embedding_dtype = _load_like_adapter(checkpoint, _encoder_config())
    model.prepare(
        "cpu",
        use_flash=True,
        compute_dtype=torch.float32,
        max_positions=_WINDOW,
        embedding_dtype=embedding_dtype,
        fused_rope=fused_rope,
    )
    return model


def _adapter(model: LayaModel, *, cuda_graphs: str = "off") -> LayaAdapter:
    adapter = LayaAdapter("tiny", max_seq_length=_WINDOW, cuda_graphs=cuda_graphs)
    adapter._model, adapter._device, adapter._max_len = model, "cpu", _WINDOW
    adapter._tokenizer = SimpleNamespace(pad_token_id=PAD_ID)  # ty: ignore[invalid-assignment]
    return adapter


def _pair(checkpoint: Path) -> tuple[LayaAdapter, LayaAdapter, _Runner]:
    """One model behind two adapters: graphs replayed on the CPU, and eager."""
    model = _flash_model(checkpoint)
    graphed, eager = _adapter(model, cuda_graphs="bucketed"), _adapter(model)
    encode, tables = model.graph_encoder()
    runner = _Runner(
        encode, hidden_size=model.encoder.config.hidden_size, window=_WINDOW, max_tokens=256, static_tensors=tables
    )
    graphed._graphs, graphed._graph_tokens = runner, 256
    return graphed, eager, runner


def _assert_rows_close(got: list[np.ndarray], want: list[np.ndarray]) -> None:
    assert len(got) == len(want) == len(ROWS)
    for mine, ref, markers in zip(got, want, MARKERS, strict=True):
        assert mine.dtype == np.float32
        assert mine.shape == (len(markers),)
        assert np.isfinite(mine).all()
        np.testing.assert_allclose(mine, ref, atol=_TOLERANCE, rtol=_TOLERANCE)
        # Guard against a vacuous pass: the options get distinct scores.
        assert float(ref.max() - ref.min()) > 1e-3


@pytest.mark.usefixtures("reference_flash")
class TestForward:
    def test_replayed_rows_match_the_eager_forward(self, reference_checkpoint: tuple[Any, Path]) -> None:  # noqa: F811
        graphed, eager, runner = _pair(reference_checkpoint[1])
        want = eager._run_rows(ROWS, _prefixes())
        _assert_rows_close(graphed._run_rows(ROWS, _prefixes()), want)
        _assert_rows_close(graphed._run_rows(ROWS, _prefixes()), want)
        # 71 packed tokens in three rows: one 128-token graph, recorded once and replayed after.
        assert runner.recorded == [(128, 64)]
        assert (runner.stats.recorded, runner.stats.replayed, runner.stats.eager) == (1, 1, {})

    def test_the_graph_records_the_encoder_the_eager_forward_runs(
        self,
        reference_checkpoint: tuple[Any, Path],  # noqa: F811
    ) -> None:
        """``encode_packed`` on a padded stream gives the real rows what the packed eager stream gives them."""
        model = _flash_model(reference_checkpoint[1])
        lengths = [len(row) for row in ROWS]
        total = sum(lengths)
        ids = torch.tensor([token for row in ROWS for token in row], dtype=torch.int32)
        cu = torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), dtype=torch.int32)
        positions = torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths])
        with torch.inference_mode():
            packed = model.encode_packed(ids, positions, cu, max(lengths), total)
            # As a 128-token graph lays it out: padding tokens that belong to no
            # sequence, and five empty sequence slots after the three rows.
            extra = 128 - total
            padded_ids = torch.cat([ids, torch.full((extra,), PAD_ID, dtype=torch.int32)])
            padded_positions = torch.cat([positions, torch.zeros(extra, dtype=torch.int32)])
            padded_cu = torch.cat([cu, torch.full((5,), total, dtype=torch.int32)])
            padded = model.encode_packed(padded_ids, padded_positions, padded_cu, _WINDOW, 128)
        assert packed.dtype == padded.dtype == torch.float32
        torch.testing.assert_close(padded[:total], packed, atol=_TOLERANCE, rtol=_TOLERANCE)

    def test_a_forward_past_the_bound_runs_eagerly(self, reference_checkpoint: tuple[Any, Path]) -> None:  # noqa: F811
        graphed, eager, runner = _pair(reference_checkpoint[1])
        runner._max_tokens, runner._tokens = 64, (64,)
        _assert_rows_close(graphed._run_rows(ROWS, _prefixes()), eager._run_rows(ROWS, _prefixes()))
        assert runner.stats.eager == {"too_large": 1}
        assert runner.recorded == []

    def test_a_request_past_the_adapters_bound_is_not_offered_to_the_runner(
        self,
        reference_checkpoint: tuple[Any, Path],  # noqa: F811
    ) -> None:
        """The GPU bounds a large request, so none of its forwards replays, however small the last one is."""
        graphed, eager, runner = _pair(reference_checkpoint[1])
        graphed._graph_tokens = 64  # the three rows hold 71 tokens
        graphed._max_forward_tokens = eager._max_forward_tokens = 40  # one row per forward, each within a graph
        _assert_rows_close(graphed._run_rows(ROWS, _prefixes()), eager._run_rows(ROWS, _prefixes()))
        assert runner.stats.forwards == 0

    def test_chunks_replay_one_at_a_time(self, reference_checkpoint: tuple[Any, Path]) -> None:  # noqa: F811
        """A request split into several forwards replays each chunk that fits a graph."""
        graphed, eager, runner = _pair(reference_checkpoint[1])
        graphed._max_forward_tokens = eager._max_forward_tokens = 40  # one row per forward
        _assert_rows_close(graphed._run_rows(ROWS, _prefixes()), eager._run_rows(ROWS, _prefixes()))
        assert runner.stats.forwards == len(ROWS)
        assert runner.stats.eager == {}

    def test_unload_drops_the_graphs(self, reference_checkpoint: tuple[Any, Path]) -> None:  # noqa: F811
        graphed, _, _ = _pair(reference_checkpoint[1])
        graphed.unload()
        assert graphed._graphs is None
        assert graphed._model is None


class TestOptions:
    def test_cuda_graphs_is_a_load_time_option(self) -> None:
        reject_unknown_loadtime_options(LayaAdapter, {"cuda_graphs": "bucketed"}, model_name="m")
        assert LayaAdapter("m", cuda_graphs="bucketed")._cuda_graphs == "bucketed"
        assert LayaAdapter("m", cuda_graphs=False)._cuda_graphs == "off"
        assert LayaAdapter("m")._cuda_graphs == "off"
        with pytest.raises(ValueError, match="cuda_graphs must be 'off' or 'bucketed'"):
            LayaAdapter("m", cuda_graphs="exact")

    def test_fused_rope_is_a_load_time_option(self) -> None:
        reject_unknown_loadtime_options(LayaAdapter, {"fused_rope": True}, model_name="m")
        assert LayaAdapter("m", fused_rope=True)._fused_rope is True
        assert LayaAdapter("m")._fused_rope is False
        with pytest.raises(ValueError, match="fused_rope must be true or false"):
            LayaAdapter("m", fused_rope="yes")  # ty: ignore[invalid-argument-type]

    def test_an_adapter_without_graphs_builds_no_runner(self) -> None:
        adapter = LayaAdapter("m")
        adapter._model, adapter._device = SimpleNamespace(use_flash=True), "cuda:0"  # ty: ignore[invalid-assignment]
        adapter._tokenizer = SimpleNamespace(pad_token_id=PAD_ID)  # ty: ignore[invalid-assignment]
        assert adapter._graph_runner() is None

    def test_off_cuda_the_adapter_builds_no_runner(
        self,
        reference_checkpoint: tuple[Any, Path],  # noqa: F811
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        adapter = _adapter(_flash_model(reference_checkpoint[1]), cuda_graphs="bucketed")
        with caplog.at_level(logging.WARNING):
            assert adapter._graph_runner() is None
        assert "CUDA graphs need a CUDA device" in caplog.text

    @pytest.mark.parametrize(("hidden_size", "tokens"), [(384, 2048), (768, 2048), (1024, 1024)])
    def test_the_bound_depends_on_the_width_and_is_a_bucket(self, hidden_size: int, tokens: int) -> None:
        assert graph_token_bound(hidden_size) == tokens
        assert token_buckets(tokens)[-1] == tokens

    def test_shipped_checkpoints_offer_graphs_as_their_own_profile(self) -> None:
        """Graphs are opt-in: the default profile runs eagerly, and ``MODEL:graphs`` is the same model with them on."""
        configs = load_model_configs(_MODELS_DIR)
        served = {
            config.sie_id
            for config in configs.values()
            if any(config.resolve_profile(name).adapter_path == _LAYA_ADAPTER for name in config.profiles)
        }
        assert served == _SHIPPED | {f"{sie_id}:graphs" for sie_id in _SHIPPED}
        for sie_id in _SHIPPED:
            default = configs[sie_id].resolve_profile("default")
            assert "cuda_graphs" not in default.loadtime
            graphs = configs[f"{sie_id}:graphs"].resolve_profile("default")
            reject_unknown_loadtime_options(LayaAdapter, dict(graphs.loadtime), model_name=f"{sie_id}:graphs")
            assert parse_graph_mode(graphs.loadtime.get("cuda_graphs"), adapter=_LAYA_ADAPTER) == "bucketed"
            assert (graphs.adapter_path, graphs.compute_precision, graphs.max_batch_tokens) == (
                default.adapter_path,
                default.compute_precision,
                default.max_batch_tokens,
            )
            assert configs[f"{sie_id}:graphs"].hf_revision == configs[sie_id].hf_revision


def _cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    from sie_server.core.inference import is_flash_attention_available

    return is_flash_attention_available("cuda:0")


@pytest.mark.gpu_hw
class TestOnGpu:
    def test_recorded_graphs_match_eager(self, reference_checkpoint: tuple[Any, Path]) -> None:  # noqa: F811
        if not _cuda_ready():
            pytest.skip("requires CUDA and flash-attn")
        adapters = []
        for graphs in ("bucketed", "off"):
            model, embedding_dtype = _load_like_adapter(reference_checkpoint[1], _encoder_config())
            model.prepare(
                "cuda:0",
                use_flash=True,
                compute_dtype=torch.bfloat16,
                max_positions=_WINDOW,
                embedding_dtype=embedding_dtype,
            )
            adapter = LayaAdapter("tiny", max_seq_length=_WINDOW, cuda_graphs=graphs)
            adapter._model, adapter._device, adapter._max_len = model, "cuda:0", _WINDOW
            adapter._tokenizer = SimpleNamespace(pad_token_id=PAD_ID)  # ty: ignore[invalid-assignment]
            adapter._graphs = adapter._graph_runner()
            adapters.append(adapter)
        graphed, eager = adapters
        assert graphed._graphs is not None
        assert eager._graphs is None
        for _ in range(3):
            got = graphed._run_rows(ROWS, _prefixes())
            want = eager._run_rows(ROWS, _prefixes())
            for mine, ref in zip(got, want, strict=True):
                # bfloat16 projections: the graph pads the packed stream, which moves sums by rounding.
                np.testing.assert_allclose(mine, ref, atol=5e-2)
        stats = graphed._graphs.stats
        assert stats.recorded == 1
        assert stats.replayed == 2
        assert stats.recording_failures == 0
        assert graphed._graphs.held_bytes > 0
