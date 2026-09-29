"""TopK-Embed's warm-up: every tuning step of the packed path, then a query and a page, on CUDA only."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter, _Launched
from sie_server.types.inputs import decode_image

from .test_topk_embed import make_adapter
from .test_topk_embed_packed import _IdentityText


def _packed(**kwargs: Any) -> TopkEmbedAdapter:
    adapter = make_adapter(packed=True, **kwargs)
    adapter._packed_text = cast("Any", _IdentityText())
    return adapter


def _warm(adapter: TopkEmbedAdapter, *, oom_from: int | None = None) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Run the warm-up as on CUDA; returns each packed batch's row lengths and each ``encode`` call."""
    batches: list[list[int]] = []
    encodes: list[dict[str, Any]] = []

    def forward(rows: list[torch.Tensor], **kwargs: Any) -> _Launched:
        assert kwargs["graphs"] is False
        batches.append([len(row) for row in rows])
        if oom_from is not None and sum(batches[-1]) >= oom_from:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return _Launched(torch.empty(0), None, lambda _: [])

    def encode(items: list[Any], output_types: list[str], **kwargs: Any) -> None:
        encodes.append({"items": items, **kwargs})

    adapter._device = "cuda:0"
    with (
        patch.object(adapter, "_forward_packed", side_effect=forward),
        patch.object(adapter, "encode", side_effect=encode),
    ):
        adapter.warmup()
    return batches, encodes


def test_one_packed_batch_per_step_up_to_the_largest_batch() -> None:
    adapter = _packed(text_batch_tokens=4096)
    batches, _ = _warm(adapter)
    assert [sum(batch) for batch in batches] == [1024, 2048, 3072, 4096]
    # Rows stay within the document cap, as real batches do.
    assert all(length <= adapter._doc_max_length for batch in batches for length in batch)


def test_the_steps_follow_the_per_head_norms() -> None:
    # 8 query heads: the attention layers' norms see 8 rows per token, so they tune every 256 tokens.
    adapter = _packed(text_batch_tokens=1024)
    adapter._attention_heads = 8
    batches, _ = _warm(adapter)
    assert [sum(batch) for batch in batches] == [256, 512, 768, 1024]


def test_the_graphs_padded_forward_warms_at_each_convolution_step() -> None:
    adapter = _packed(text_batch_tokens=1024)
    graphs = MagicMock()
    adapter._graphs = graphs
    _warm(adapter)
    graphs.warm_up.assert_called_once_with(1024)


def test_a_batch_of_pages_can_be_the_largest() -> None:
    adapter = _packed(text_batch_tokens=1024, image_batch_size=64)
    page = 16 + len(adapter._image_prefix) + len(adapter._image_suffix)  # 16 merged patches at most
    assert adapter._largest_batch_tokens() == 64 * page
    batches, _ = _warm(adapter)
    assert sum(batches[-2]) < 64 * page <= sum(batches[-1])


def test_then_a_query_and_a_page() -> None:
    _, encodes = _warm(_packed(text_batch_tokens=1024))
    query, page = encodes
    assert query["is_query"] is True
    assert query["items"][0].text
    assert not page.get("is_query", False)
    (image,) = page["items"][0].images
    assert decode_image(image).size == (256, 256)


def test_out_of_memory_stops_the_steps_but_not_the_load() -> None:
    batches, encodes = _warm(_packed(text_batch_tokens=4096), oom_from=3072)
    assert [sum(batch) for batch in batches] == [1024, 2048, 3072]
    assert len(encodes) == 2


def test_other_errors_fail_the_load() -> None:
    adapter = _packed(text_batch_tokens=2048)
    adapter._device = "cuda:0"
    with (
        patch.object(adapter, "_forward_packed", side_effect=RuntimeError("an illegal memory access")),
        pytest.raises(RuntimeError, match="illegal memory access"),
    ):
        adapter.warmup()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_nothing_to_compile_off_cuda(device: str) -> None:
    adapter = _packed()
    adapter._device = device
    with patch.object(adapter, "_forward_packed") as forward, patch.object(adapter, "encode") as encode:
        adapter.warmup()
    forward.assert_not_called()
    encode.assert_not_called()


def test_nothing_to_compile_on_the_row_per_input_path() -> None:
    adapter = make_adapter()
    adapter._device = "cuda:0"
    with patch.object(adapter, "_forward_packed") as forward, patch.object(adapter, "encode") as encode:
        adapter.warmup()
    forward.assert_not_called()
    encode.assert_not_called()


def test_the_warm_up_forward_skips_the_graphs() -> None:
    adapter = _packed()
    graphs = MagicMock()
    adapter._graphs = graphs
    rows = [torch.tensor([1, 2, 3])]
    launched = adapter._forward_packed(rows, images=None, normalize=True, dtype=torch.float32, graphs=False)
    assert [len(row) for row in launched.rows()] == [3]
    graphs.run.assert_not_called()
