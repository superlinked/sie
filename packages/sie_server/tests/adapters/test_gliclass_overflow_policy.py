from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.gliclass import GLiClassAdapter


class _FakeTokenizer:
    def __call__(self, text: str | list[str], add_special_tokens: bool = False) -> dict[str, Any]:
        if isinstance(text, list):
            return {"input_ids": [t.split() for t in text]}
        return {"input_ids": text.split()}

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(ids)


class _FakePipe:
    def prepare_input(self, text: str, labels: list[str]) -> str:
        prefix = "<LABEL> " + " <SEP> ".join(labels)
        return f"{prefix} {text}".strip()


# With _FakeTokenizer + _FakePipe, N labels produce a 2N-token label_prompt.
# _LABELS → "<LABEL> x <SEP> y" = 4 tokens; with special_count=2 the overhead
# is 6, so for max_seq_length=10 the per-text budget is 4 tokens.
_LABELS = ["x", "y"]


def _make_adapter(*, max_seq_length: int = 10, special_count: int = 2) -> GLiClassAdapter:
    adapter = GLiClassAdapter("test-model")
    adapter._max_seq_length = max_seq_length
    adapter._special_count = special_count
    adapter._tokenizer = _FakeTokenizer()  # ty:ignore[invalid-assignment]
    adapter._pipe = _FakePipe()
    return adapter


class TestApplyOverflowPolicy:
    def test_default_is_noop_even_when_input_overflows(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e f g h i j"]  # 10 tokens, observed = 16 > 10

        assert adapter._apply_overflow_policy(texts, _LABELS, "default") == texts

    def test_default_is_the_default_arg(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e f g h i j"]

        assert adapter._apply_overflow_policy(texts, _LABELS) == texts

    def test_truncate_text_passes_fitting_text_through(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d"]  # 4 tokens, observed = 10

        assert adapter._apply_overflow_policy(texts, _LABELS, "truncate_text") == ["a b c d"]

    def test_truncate_text_slices_overflowing_text_to_budget(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e f g"]  # 7 tokens, budget = 4

        assert adapter._apply_overflow_policy(texts, _LABELS, "truncate_text") == ["a b c d"]

    def test_truncate_text_mixed_batch(self) -> None:
        adapter = _make_adapter()
        texts = ["a b", "a b c d e f g"]

        assert adapter._apply_overflow_policy(texts, _LABELS, "truncate_text") == ["a b", "a b c d"]

    def test_error_raises_on_overflowing_text(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e"]  # 5 tokens, observed = 11 > 10

        with pytest.raises(InputTooLongError, match=r"items\[0\] observed_tokens=11"):
            adapter._apply_overflow_policy(texts, _LABELS, "error")

    def test_error_reports_first_overflowing_item_index(self) -> None:
        adapter = _make_adapter()
        texts = ["a b", "a b c d e f"]

        with pytest.raises(InputTooLongError, match=r"items\[1\]"):
            adapter._apply_overflow_policy(texts, _LABELS, "error")

    def test_label_prompt_overflow_raises_under_truncate_text(self) -> None:
        adapter = _make_adapter(max_seq_length=5)  # overhead 6 > 5

        with pytest.raises(InputTooLongError, match="label_prompt"):
            adapter._apply_overflow_policy(["a"], _LABELS, "truncate_text")

    def test_label_prompt_overflow_raises_under_error(self) -> None:
        adapter = _make_adapter(max_seq_length=5)

        with pytest.raises(InputTooLongError, match="label_prompt"):
            adapter._apply_overflow_policy(["a"], _LABELS, "error")

    def test_label_prompt_overflow_does_not_raise_under_default(self) -> None:
        adapter = _make_adapter(max_seq_length=5)
        texts = ["a b c"]

        assert adapter._apply_overflow_policy(texts, _LABELS, "default") == texts


class _RaisingPipe:
    """Fake gliclass pipe whose model input preparation raises a chosen exception.

    Exercises the crash-signature mapping in ``GLiClassAdapter.extract``
    end-to-end (not just ``_apply_overflow_policy``). The default overflow
    policy returns texts without tokenizing, so the pipe only needs to fail
    when the rows are prepared.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def prepare_inputs(self, *args: object, **kwargs: object) -> object:
        raise self._exc


def _make_raising_adapter(exc: BaseException) -> GLiClassAdapter:
    adapter = GLiClassAdapter("test-model")
    adapter._pipe = _RaisingPipe(exc)
    return adapter


class TestExtractOverflowGuard:
    """The ``extract`` except block maps known crash signatures to
    ``InputTooLongError`` and lets unrelated errors propagate unchanged.
    """

    def test_label_window_off_by_one_maps_to_input_too_long(self) -> None:
        # The new #1434 case: too many labels overflow the shared window, only
        # ~79 of 100 survive, and the single-label decode indexes past the
        # shrunk window (size > 0).
        adapter = _make_raising_adapter(IndexError("index 79 is out of bounds for dimension 0 with size 79"))
        items = [SimpleNamespace(text="hello world")]

        with pytest.raises(InputTooLongError):
            adapter.extract(items, labels=_LABELS)  # ty:ignore[invalid-argument-type]

    def test_empty_tensor_index_maps_to_input_too_long(self) -> None:
        adapter = _make_raising_adapter(IndexError("index 0 is out of bounds for dimension 0 with size 0"))
        items = [SimpleNamespace(text="hello world")]

        with pytest.raises(InputTooLongError):
            adapter.extract(items, labels=_LABELS)  # ty:ignore[invalid-argument-type]

    def test_argmax_empty_tensor_maps_to_input_too_long(self) -> None:
        adapter = _make_raising_adapter(
            RuntimeError("argmax(): Expected reduction dim to be specified for input.numel() == 0.")
        )
        items = [SimpleNamespace(text="hello world")]

        with pytest.raises(InputTooLongError):
            adapter.extract(items, labels=_LABELS)  # ty:ignore[invalid-argument-type]

    def test_unrelated_index_error_propagates(self) -> None:
        adapter = _make_raising_adapter(IndexError("list index out of range"))
        items = [SimpleNamespace(text="hello world")]

        with pytest.raises(IndexError, match="list index out of range"):
            adapter.extract(items, labels=_LABELS)  # ty:ignore[invalid-argument-type]

    def test_generic_out_of_range_index_error_propagates(self) -> None:
        # index > size: a genuine out-of-range bug against a non-empty tensor,
        # NOT the #1434 off-by-one (index == size), so it must propagate.
        adapter = _make_raising_adapter(IndexError("index 5 is out of bounds for dimension 0 with size 3"))
        items = [SimpleNamespace(text="hello world")]

        with pytest.raises(IndexError, match="size 3"):
            adapter.extract(items, labels=_LABELS)  # ty:ignore[invalid-argument-type]

    def test_unrelated_runtime_error_propagates(self) -> None:
        adapter = _make_raising_adapter(RuntimeError("CUDA out of memory"))
        items = [SimpleNamespace(text="hello world")]

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            adapter.extract(items, labels=_LABELS)  # ty:ignore[invalid-argument-type]
