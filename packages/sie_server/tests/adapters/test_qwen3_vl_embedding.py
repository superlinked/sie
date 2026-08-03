from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from sie_server.adapters.qwen3_vl_embedding import (
    _DEFAULT_INSTRUCTION,
    Qwen3VLEmbeddingAdapter,
    _normalize_instruction,
)
from sie_server.core.video_frames import MAX_SAMPLED_FRAMES, VideoDecodeError
from sie_server.types.inputs import InvalidInputError


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


class _WordTokenizer:
    """Deterministic HF-shaped tokenizer: one token per word plus two specials.

    The real ``AutoProcessor`` for Qwen3-VL bundles a tokenizer, which
    ``_metering_tokenizer`` reaches through ``processor.tokenizer``; the fake
    processor above has none, so the metering tests attach this.
    """

    model_max_length = 512

    def __call__(
        self,
        text: list[str],
        text_pair: list[str] | None = None,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[list[int]]]:
        _ = text_pair
        lengths = [len(t.split()) + 2 for t in text]
        if truncation and max_length is not None:
            lengths = [min(n, max_length) for n in lengths]
        return {"input_ids": [[0] * n for n in lengths]}


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


class TestVideoFramesBillAsImages:
    """Sampled video frames are the model's real visual input, so they bill as
    §7 ``images`` on a worker-authoritative count (issue #2433).

    The count travels on ``EncodeOutput.extra["input_image_counts"]`` because
    the wire item carries only compressed bytes — nothing downstream can derive
    how many frames those bytes decoded to.
    """

    def _adapter(self) -> Qwen3VLEmbeddingAdapter:
        adapter = Qwen3VLEmbeddingAdapter("Qwen/Qwen3-VL-Embedding-2B")
        adapter._device = "cpu"
        inputs = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        adapter._model = _FakeCausalLM(torch.tensor([[[1.0, 0.0]]]))  # ty: ignore[invalid-assignment]
        adapter._processor = _FakeProcessor(inputs)  # ty: ignore[invalid-assignment]
        return adapter

    @staticmethod
    def _frames(count: int) -> list[Any]:
        return [Image.new("RGB", (4, 4)) for _ in range(count)]

    def test_video_only_item_bills_the_frames_it_processed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = self._adapter()
        monkeypatch.setattr(
            "sie_server.adapters.qwen3_vl_embedding.extract_frames",
            lambda _video: self._frames(7),
        )
        item = SimpleNamespace(text=None, images=None, video={"data": b"mp4-bytes", "format": "mp4"})

        out = adapter.encode([item], ["dense"])

        # Seven frames processed -> seven billable images, not one video blob.
        assert out.extra["input_image_counts"] == [7]

    def test_frames_reach_the_model_as_image_content_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = self._adapter()
        monkeypatch.setattr(
            "sie_server.adapters.qwen3_vl_embedding.extract_frames",
            lambda _video: self._frames(3),
        )
        item = SimpleNamespace(text="a caption", images=None, video={"data": b"mp4-bytes"})

        adapter.encode([item], ["dense"])

        processor: Any = adapter._processor
        user_turn = processor.conversations[0][1]["content"]
        assert sum(1 for part in user_turn if part["type"] == "image") == 3

    def test_mixed_images_and_frames_sum_into_one_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = self._adapter()
        monkeypatch.setattr(
            "sie_server.adapters.qwen3_vl_embedding.extract_frames",
            lambda _video: self._frames(4),
        )
        png = b"\x89PNG\r\n\x1a\n"
        monkeypatch.setattr(
            Qwen3VLEmbeddingAdapter,
            "_load_images",
            lambda _self, _item: self._frames(2),
        )
        item = SimpleNamespace(text=None, images=[{"data": png}, {"data": png}], video={"data": b"mp4"})

        out = adapter.encode([item], ["dense"])

        assert out.extra["input_image_counts"] == [6]

    def test_text_only_item_bills_no_images(self) -> None:
        adapter = self._adapter()
        item = SimpleNamespace(text="hi", images=None, video=None)
        assert adapter.encode([item], ["dense"]).extra["input_image_counts"] == [0]

    def test_per_item_counts_align_with_the_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = self._adapter()
        monkeypatch.setattr(
            "sie_server.adapters.qwen3_vl_embedding.extract_frames",
            lambda _video: self._frames(5),
        )
        items = [
            SimpleNamespace(text="text only", images=None, video=None),
            SimpleNamespace(text=None, images=None, video={"data": b"mp4"}),
        ]
        assert adapter.encode(items, ["dense"]).extra["input_image_counts"] == [0, 5]

    def test_undecodable_video_faults_instead_of_being_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: a text+video item used to embed the text alone and return
        # a billed success that silently ignored the video.
        adapter = self._adapter()

        def boom(_video: Any) -> list[Any]:
            raise VideoDecodeError("video input could not be opened by the decoder")

        monkeypatch.setattr("sie_server.adapters.qwen3_vl_embedding.extract_frames", boom)
        item = SimpleNamespace(text="a caption", images=None, video={"data": b"garbage"})

        with pytest.raises(VideoDecodeError):
            adapter.encode([item], ["dense"])

    def test_unreadable_images_are_not_billed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``_load_images`` drops an image it cannot decode with only a warning,
        # so counting SUBMITTED images would bill three for the one the model
        # actually saw — the same silent over-bill the video half forbids. Both
        # halves of the count are processed-based.
        adapter = self._adapter()
        monkeypatch.setattr(
            Qwen3VLEmbeddingAdapter,
            "_load_images",
            lambda _self, _item: self._frames(1),  # two of the three failed to load
        )
        png = b"\x89PNG\r\n\x1a\n"
        item = SimpleNamespace(text=None, images=[{"data": png}] * 3, video=None)

        out = adapter.encode([item], ["dense"])

        assert out.extra["input_image_counts"] == [1]

    def test_the_processed_count_never_exceeds_the_reserved_basis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The gateway reserves `submitted_images + videos * 32`. A count built
        # from loaded images plus sampled frames is bounded by that basis, so it
        # can never push settlement over its ceiling.
        adapter = self._adapter()
        monkeypatch.setattr(
            "sie_server.adapters.qwen3_vl_embedding.extract_frames",
            lambda _video: self._frames(MAX_SAMPLED_FRAMES),
        )
        monkeypatch.setattr(
            Qwen3VLEmbeddingAdapter,
            "_load_images",
            lambda _self, _item: self._frames(1),
        )
        png = b"\x89PNG\r\n\x1a\n"
        item = SimpleNamespace(text=None, images=[{"data": png}] * 2, video={"data": b"mp4"})

        processed = adapter.encode([item], ["dense"]).extra["input_image_counts"][0]

        reserved = len(item.images) + 1 * MAX_SAMPLED_FRAMES
        assert processed <= reserved
        assert processed == 1 + MAX_SAMPLED_FRAMES

    def test_token_counts_scatter_across_a_mixed_batch(self) -> None:
        # The base hook is all-or-nothing: one non-text item and the WHOLE
        # batch loses its token counts, which left the text item with no units
        # at all and faulted the dispatch at the gateway. Video items are
        # non-text by construction and the queue seam fuses requests, so this
        # adapter scatters instead — real counts for text, 0 for visual-only.
        adapter = self._adapter()
        adapter._processor.tokenizer = _WordTokenizer()  # ty: ignore[unresolved-attribute]
        items = [
            SimpleNamespace(text="alpha beta", images=None, video=None),
            SimpleNamespace(text=None, images=None, video={"data": b"mp4"}),
            SimpleNamespace(text="one two three", images=None, video=None),
        ]

        counts = adapter.count_input_tokens(items)

        assert counts is not None
        assert len(counts) == len(items)
        assert counts[1] == 0  # visual-only item contributes no text tokens
        assert counts[0] > 0
        assert counts[2] > counts[0]  # more words -> more tokens

    def test_a_video_only_batch_reports_no_text_tokens(self) -> None:
        # Nothing tokenizable: every count is zero, which ``_encode_units``
        # drops, so the request settles on ``images`` alone — matching a plan
        # that never reserved ``input_tokens`` for it.
        adapter = self._adapter()
        adapter._processor.tokenizer = _WordTokenizer()  # ty: ignore[unresolved-attribute]
        items = [SimpleNamespace(text=None, images=None, video={"data": b"mp4"})]
        assert adapter.count_input_tokens(items) == [0]

    def test_decode_failure_is_typed_invalid_input(self) -> None:
        # Both ingress paths map InvalidInputError to INVALID_INPUT / HTTP 400,
        # so an undecodable video is a 4xx and never a generic 500.
        assert issubclass(VideoDecodeError, InvalidInputError)
