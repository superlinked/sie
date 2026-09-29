"""TopK-Embed's host side: vectors copied as float16 when the response is float16, batches run back to back."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sie_server.adapters.topk_embed import adapter as adapter_module
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter, _Launched
from sie_server.adapters.topk_embed.packed import to_device
from sie_server.types.inputs import Item

from .test_topk_embed import _fake_vision, _png, make_adapter
from .test_topk_embed_packed import _IdentityText

TEXTS = [Item(text="a short one"), Item(text="a somewhat longer document, with commas."), Item(text="x")]


@pytest.fixture(params=["padded", "packed"])
def adapter(request: pytest.FixtureRequest) -> TopkEmbedAdapter:
    if request.param == "padded":
        return make_adapter()
    packed_adapter = make_adapter(packed=True)
    packed_adapter._packed_text = cast("Any", _IdentityText())
    return packed_adapter


class TestTransferDtype:
    def test_float16_responses_come_back_as_float16(self, adapter: TopkEmbedAdapter) -> None:
        full = adapter.encode(TEXTS, ["multivector"])
        half = adapter.encode(TEXTS, ["multivector"], options={"output_dtype": "float16"})
        assert full.multivector is not None
        assert half.multivector is not None
        for a, b in zip(full.multivector, half.multivector, strict=True):
            assert a.dtype == np.float32
            assert b.dtype == np.float16
            np.testing.assert_allclose(b.astype(np.float32), a, rtol=0, atol=1e-3)

    @pytest.mark.parametrize(
        "options",
        [{}, {"output_dtype": "float32"}, {"output_dtype": "int8"}, {"output_dtype": "float16", "muvera": {}}],
    )
    def test_everything_else_stays_float32(self, adapter: TopkEmbedAdapter, options: dict[str, Any]) -> None:
        # MUVERA reads the multi-vectors after the adapter, so they stay at full precision.
        output = adapter.encode(TEXTS, ["multivector"], options=options)
        assert output.multivector is not None
        assert all(vector.dtype == np.float32 for vector in output.multivector)

    def test_page_vectors_follow_the_same_rule(self, adapter: TopkEmbedAdapter) -> None:
        with _fake_vision(adapter):
            half = adapter.encode([Item(images=[_png()])], ["multivector"], options={"output_dtype": "float16"})
        assert half.multivector is not None
        assert half.multivector[0].dtype == np.float16

    def test_scores_use_float32_whatever_the_response(self, adapter: TopkEmbedAdapter) -> None:
        options = {"output_dtype": "float16"}
        scores = adapter.score(Item(text="a short one"), TEXTS, options=options)
        reference = adapter.score(Item(text="a short one"), TEXTS)
        assert scores == reference


class TestBackToBack:
    def test_each_batch_is_unpacked_after_the_next_one_launches(self) -> None:
        adapter = make_adapter()
        events: list[str] = []

        def launch(rows: list[torch.Tensor], **kwargs: Any) -> _Launched:
            index = len([e for e in events if e.startswith("launch")])
            events.append(f"launch {index}")
            return _Launched(torch.empty(0), None, lambda _: [torch.zeros(len(row), 6) for row in rows])

        def deliver(batch: list[int], launched: _Launched) -> None:
            events.append(f"deliver {batch[0]}")
            launched.rows()

        batches = [([i], [torch.tensor([1, 2, 3])], None) for i in range(3)]
        with patch.object(adapter, "_launch", side_effect=launch):
            adapter._run_batches(batches, normalize=True, dtype=torch.float32, deliver=deliver)
        assert events == ["launch 0", "launch 1", "deliver 0", "launch 2", "deliver 1", "deliver 2"]

    def test_many_batches_give_the_same_vectors(self, adapter: TopkEmbedAdapter) -> None:
        one_batch = adapter.encode(TEXTS, ["multivector"])
        adapter._text_batch_tokens = 1  # one row per batch
        many = adapter.encode(TEXTS, ["multivector"])
        assert one_batch.multivector is not None
        assert many.multivector is not None
        for a, b in zip(one_batch.multivector, many.multivector, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_a_ready_launch_hands_back_its_rows(self) -> None:
        launched = _Launched(torch.arange(6.0).view(2, 3), None, list)
        assert [row.tolist() for row in launched.rows()] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_host_copies_stay_plain_off_cuda() -> None:
    tensor = torch.arange(4)
    moved = to_device(tensor, "cpu")
    assert moved.device.type == "cpu"
    assert torch.equal(moved, tensor)


def test_the_adapter_module_keeps_its_forward_entry_points() -> None:
    assert callable(adapter_module.TopkEmbedAdapter._launch)
    assert callable(adapter_module.TopkEmbedAdapter._run_batches)
