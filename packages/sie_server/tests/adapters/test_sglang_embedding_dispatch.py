"""SGLangEmbeddingAdapter can let the worker keep several batches in flight, with metering still exact."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.loader import load_model_config
from sie_server.core.worker.model_worker import _dispatch_width

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def test_one_batch_at_a_time_unless_a_profile_asks() -> None:
    adapter = SGLangEmbeddingAdapter("test-model")
    assert adapter.max_concurrent_dispatch() == 1
    assert _dispatch_width(adapter) == 1


def test_qwen3_embedding_4b_keeps_four_batches_in_flight() -> None:
    loadtime = load_model_config(_MODELS_DIR / "Qwen__Qwen3-Embedding-4B.yaml").resolve_profile("default").loadtime
    assert loadtime["max_concurrent_dispatch"] == 4
    adapter = SGLangEmbeddingAdapter("Qwen/Qwen3-Embedding-4B", **loadtime)
    assert _dispatch_width(adapter) == 4


def test_lora_keeps_one_batch_at_a_time() -> None:
    adapter = SGLangEmbeddingAdapter("test-model", max_concurrent_dispatch=4, lora_paths={"legal": "org/legal-lora"})
    assert adapter.max_concurrent_dispatch() == 4
    assert _dispatch_width(adapter) == 1


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "4"])
def test_invalid_dispatch_width_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="max_concurrent_dispatch"):
        SGLangEmbeddingAdapter("test-model", max_concurrent_dispatch=value)


class _NonReentrantTokenizer:
    """Fails like a fast tokenizer ("Already borrowed") when two threads use it at once."""

    def __init__(self) -> None:
        self._busy = threading.Lock()

    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        if not self._busy.acquire(blocking=False):
            msg = "Already borrowed"
            raise RuntimeError(msg)
        try:
            time.sleep(0.002)
            return {"input_ids": [[0] * len(text.split()) for text in texts]}
        finally:
            self._busy.release()


def test_concurrent_batches_are_all_metered() -> None:
    adapter = SGLangEmbeddingAdapter("test-model", max_concurrent_dispatch=4)
    adapter._metering_tokenizer_obj = _NonReentrantTokenizer()
    adapter._metering_tokenizer_loaded = True

    def stamp(k: int) -> list[int] | None:
        output = EncodeOutput(dense=None, batch_size=2)
        adapter._stamp_input_token_counts(output, ["a b", "c d e" * (k % 3 + 1)], [0, 1], 2)
        return output.extra.get("input_token_counts")

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(stamp, range(64)))
    assert all(found is not None for found in counts)
