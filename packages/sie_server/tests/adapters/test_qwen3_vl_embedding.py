from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sie_server.adapters.qwen3_vl_embedding import (
    _DEFAULT_INSTRUCTION,
    Qwen3VLEmbeddingAdapter,
    _normalize_instruction,
)


class TestNormalizeInstruction:
    """Match the official ``format_model_input`` instruction shaping.

    The reference recipe strips the instruction and appends ``.`` unless the
    final character is already Unicode punctuation. SIE previously passed the
    instruction verbatim, so MTEB query instructions without trailing
    punctuation differed from the official prompt by a missing period token.
    """

    def test_appends_period_when_missing(self) -> None:
        assert (
            _normalize_instruction("Given a financial question, retrieve user replies that best answer the question")
            == "Given a financial question, retrieve user replies that best answer the question."
        )

    def test_keeps_existing_trailing_period(self) -> None:
        assert _normalize_instruction("Represent the user's input.") == "Represent the user's input."

    def test_keeps_other_trailing_punctuation(self) -> None:
        # '?', '!', ':' are all Unicode category 'P*' -> no extra period.
        assert _normalize_instruction("What is the capital?") == "What is the capital?"
        assert _normalize_instruction("Find the answer!") == "Find the answer!"
        assert _normalize_instruction("Retrieve documents:") == "Retrieve documents:"

    def test_strips_surrounding_whitespace(self) -> None:
        assert _normalize_instruction("  retrieve relevant passages  ") == "retrieve relevant passages."

    def test_strips_then_keeps_trailing_punctuation(self) -> None:
        assert _normalize_instruction("  Answer the query.  ") == "Answer the query."

    def test_empty_stays_empty(self) -> None:
        assert _normalize_instruction("") == ""
        assert _normalize_instruction("   ") == ""

    def test_default_instruction_is_noop(self) -> None:
        # The default already ends in punctuation -> documents are unaffected.
        assert _normalize_instruction(_DEFAULT_INSTRUCTION) == _DEFAULT_INSTRUCTION


class _FakeBaseModel:
    """Stand-in for ``Qwen3VLModel`` exposing a post-RMSNorm ``last_hidden_state``."""

    def __init__(self, last_hidden: torch.Tensor) -> None:
        self._last_hidden = last_hidden
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=self._last_hidden)


class _FakeCausalLM:
    """Stand-in for ``Qwen3VLForConditionalGeneration``.

    Exposes ``.model`` (the base ``Qwen3VLModel``) and asserts that the adapter
    never calls the CausalLM wrapper directly (which would return PRE-norm
    per-layer ``hidden_states`` instead of the post-norm ``last_hidden_state``).
    """

    def __init__(self, last_hidden: torch.Tensor) -> None:
        self.model = _FakeBaseModel(last_hidden)

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        raise AssertionError(
            "adapter must pool from self._model.model(...).last_hidden_state, not the CausalLM wrapper output"
        )


class _FakeProcessor:
    def __init__(self, inputs: dict[str, torch.Tensor]) -> None:
        self._inputs = inputs
        self.conversations: list[Any] = []

    def apply_chat_template(self, conversation: Any = None, *_args: Any, **_kwargs: Any) -> str:
        self.conversations.append(conversation)
        return "PROMPT"

    def __call__(self, **_kwargs: Any) -> dict[str, torch.Tensor]:
        return self._inputs


class _RaceDetectingProcessor(_FakeProcessor):
    """Emulate a fast tokenizer that raises on concurrent mutation."""

    def __init__(self, inputs: dict[str, torch.Tensor]) -> None:
        super().__init__(inputs)
        self._active = 0
        self._guard = threading.Lock()
        self.peak_concurrency = 0

    def _enter(self) -> bool:
        with self._guard:
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
            return self._active > 1

    def _exit(self) -> None:
        with self._guard:
            self._active -= 1

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        concurrent = self._enter()
        try:
            if concurrent:
                raise RuntimeError("Already borrowed")
            time.sleep(0.002)
            return super().apply_chat_template(*args, **kwargs)
        finally:
            self._exit()

    def __call__(self, **_kwargs: Any) -> dict[str, torch.Tensor]:
        concurrent = self._enter()
        try:
            if concurrent:
                raise RuntimeError("Already borrowed")
            time.sleep(0.002)
            return self._inputs
        finally:
            self._exit()


class TestPostNormPooling:
    """The forward path must pool the post-RMSNorm ``last_hidden_state``."""

    @pytest.fixture
    def adapter(self) -> Qwen3VLEmbeddingAdapter:
        a = Qwen3VLEmbeddingAdapter("Qwen/Qwen3-VL-Embedding-2B")
        a._device = "cpu"
        return a

    def test_pools_last_token_from_last_hidden_state(self, adapter: Qwen3VLEmbeddingAdapter) -> None:
        # seq_len=3, hidden_dim=4; attention mask marks all 3 tokens valid.
        last_hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]])
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        fake_model = _FakeCausalLM(last_hidden)
        adapter._model = fake_model  # ty: ignore[invalid-assignment]
        adapter._processor = _FakeProcessor(inputs)  # ty: ignore[invalid-assignment]

        result = adapter._forward_conversation([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])

        # Last-token vector [3, 4, 0, 0] L2-normalized -> [0.6, 0.8, 0, 0].
        assert result.shape == (4,)
        assert pytest.approx(result.tolist(), abs=1e-5) == [0.6, 0.8, 0.0, 0.0]
        # The base model was called (post-norm path), not the CausalLM wrapper.
        assert len(fake_model.model.calls) == 1
        assert "output_hidden_states" not in fake_model.model.calls[0]

    def test_mean_pool_uses_last_hidden_state(self, adapter: Qwen3VLEmbeddingAdapter) -> None:
        adapter._pooling = "mean"
        last_hidden = torch.tensor([[[2.0, 0.0], [4.0, 0.0]]])
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        adapter._model = _FakeCausalLM(last_hidden)  # ty: ignore[invalid-assignment]
        adapter._processor = _FakeProcessor(inputs)  # ty: ignore[invalid-assignment]

        result = adapter._forward_conversation([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])

        # mean([2,0],[4,0]) = [3,0] -> normalized [1,0].
        assert result.shape == (2,)
        assert pytest.approx(result.tolist(), abs=1e-5) == [1.0, 0.0]

    def test_processor_tokenization_is_thread_safe(self, adapter: Qwen3VLEmbeddingAdapter) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        processor = _RaceDetectingProcessor(inputs)
        adapter._model = _FakeCausalLM(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))  # ty: ignore[invalid-assignment]
        adapter._processor = processor  # ty: ignore[invalid-assignment]
        conversation = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: adapter._forward_conversation(conversation), range(32)))

        assert len(results) == 32
        assert processor.peak_concurrency == 1


class TestInstructionResolution:
    """``encode()`` resolves the system-turn instruction.

    The official recipe always uses a non-empty system instruction, so both an
    omitted (``None``) and an empty (``""``) instruction coalesce to the model
    default; a non-empty instruction is forwarded after ``_normalize_instruction``
    shaping. This is the inverse of preserving ``""`` as an empty system turn,
    which the model was never trained on.
    """

    def _run(self, instruction: str | None) -> str:
        adapter = Qwen3VLEmbeddingAdapter("Qwen/Qwen3-VL-Embedding-2B")
        adapter._device = "cpu"
        inputs = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        proc = _FakeProcessor(inputs)
        adapter._model = _FakeCausalLM(torch.tensor([[[1.0, 0.0]]]))  # ty: ignore[invalid-assignment]
        adapter._processor = proc  # ty: ignore[invalid-assignment]
        item = SimpleNamespace(text="hi", images=None, video=None)
        adapter.encode([item], ["dense"], instruction=instruction)
        # conversation[0] is the system turn: {"role": "system", "content": [{"type": "text", "text": ...}]}
        return proc.conversations[0][0]["content"][0]["text"]

    def test_none_uses_default(self) -> None:
        assert self._run(None) == _DEFAULT_INSTRUCTION

    def test_empty_string_coalesces_to_default(self) -> None:
        # CodeRabbit suggested preserving "" as a distinct value; for this model
        # "" is not a trained input, so it must resolve to the default instead.
        assert self._run("") == _DEFAULT_INSTRUCTION

    def test_whitespace_only_coalesces_to_default(self) -> None:
        # Whitespace-only is truthy but normalizes to "" -> must fall back to the
        # default rather than forwarding an empty system turn.
        assert self._run("   ") == _DEFAULT_INSTRUCTION

    def test_non_empty_is_normalized_and_forwarded(self) -> None:
        assert self._run("Find relevant passages") == "Find relevant passages."
