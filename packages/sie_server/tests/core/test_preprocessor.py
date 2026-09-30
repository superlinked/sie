"""Tests for preprocessor implementations."""

import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from sie_server.core.prepared import ImagePayload, PreparedItem, TextPayload
from sie_server.core.preprocessor import ImagePreprocessor, Preprocessor, TextPreprocessor
from sie_server.core.preprocessor.image import OpenCLIPImagePreprocessor
from sie_server.core.preprocessor.vision import DetectionPreprocessor
from sie_server.types.inputs import ImageInput, InvalidMediaError, Item


class TestPreprocessorProtocol:
    """Tests for Preprocessor protocol."""

    def test_text_preprocessor_is_preprocessor(self):
        """TextPreprocessor satisfies Preprocessor protocol."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        preprocessor = TextPreprocessor(tokenizer, "test-model")
        assert isinstance(preprocessor, Preprocessor)

    def test_image_preprocessor_is_preprocessor(self):
        """ImagePreprocessor satisfies Preprocessor protocol."""
        processor = MagicMock()
        preprocessor = ImagePreprocessor(processor, "test-model")
        assert isinstance(preprocessor, Preprocessor)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        config = MagicMock()
        config.max_sequence_length = 512
        return config

    def test_modality(self, mock_tokenizer):
        """Modality is 'text'."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        assert preprocessor.modality == "text"

    def test_prepare_single_item(self, mock_tokenizer, mock_config):
        """Prepare single text item."""
        mock_tokenizer.return_value = {
            "input_ids": [[101, 2023, 2003, 1037, 3231, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
        }

        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        items = [Item(text="This is a test")]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.modality == "text"
        assert batch.total_cost == 6  # 6 tokens

        item = batch.items[0]
        assert item.cost == 6
        assert item.original_index == 0
        assert isinstance(item.payload, TextPayload)
        assert item.payload.token_count == 6

    def test_prepare_multiple_items(self, mock_tokenizer, mock_config):
        """Prepare multiple text items."""
        mock_tokenizer.return_value = {
            "input_ids": [
                [101, 2023, 102],  # 3 tokens
                [101, 2023, 2003, 1037, 102],  # 5 tokens
                [101, 102],  # 2 tokens
            ],
            "attention_mask": [
                [1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1],
            ],
        }

        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        items: list[Item] = [
            Item(text="Short"),
            Item(text="A bit longer"),
            Item(text="X"),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 3
        assert batch.total_cost == 10  # 3 + 5 + 2

        # Check original indices preserved
        indices = [item.original_index for item in batch.items]
        assert indices == [0, 1, 2]

        # Check costs
        costs = [item.cost for item in batch.items]
        assert costs == [3, 5, 2]

    def test_prepare_empty_text(self, mock_tokenizer, mock_config):
        """Handle items with None text."""
        mock_tokenizer.return_value = {
            "input_ids": [[101, 102]],
            "attention_mask": [[1, 1]],
        }

        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        items = [Item(text=None)]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        # Empty string tokenized
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == [""]  # Empty string passed

    def test_collate_single_item(self, mock_tokenizer):
        """Collate single prepared item."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")

        prepared = [
            PreparedItem(
                payload=TextPayload(
                    input_ids=[101, 2023, 102],
                    attention_mask=[1, 1, 1],
                ),
                cost=3,
                original_index=0,
            )
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)
        assert result["input_ids"].tolist() == [[101, 2023, 102]]

    def test_collate_with_padding(self, mock_tokenizer):
        """Collate pads shorter sequences."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")

        prepared = [
            PreparedItem(
                payload=TextPayload(
                    input_ids=[101, 2023, 102],
                    attention_mask=[1, 1, 1],
                ),
                cost=3,
                original_index=0,
            ),
            PreparedItem(
                payload=TextPayload(
                    input_ids=[101, 102],
                    attention_mask=[1, 1],
                ),
                cost=2,
                original_index=1,
            ),
        ]

        result = preprocessor.collate(prepared, device="cpu", pad_token_id=0)

        assert result["input_ids"].shape == (2, 3)  # Padded to max length
        assert result["attention_mask"].shape == (2, 3)

        # First item unchanged
        assert result["input_ids"][0].tolist() == [101, 2023, 102]
        assert result["attention_mask"][0].tolist() == [1, 1, 1]

        # Second item padded
        assert result["input_ids"][1].tolist() == [101, 102, 0]
        assert result["attention_mask"][1].tolist() == [1, 1, 0]

    def test_collate_empty(self, mock_tokenizer):
        """Collate empty list returns empty tensors."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")

        result = preprocessor.collate([], device="cpu")

        assert result["input_ids"].numel() == 0
        assert result["attention_mask"].numel() == 0


class TestTextPreprocessorFastPath:
    """Tests for the Rust-tokenise fast-path consumer on TextPreprocessor.

    These tests cover the contract exposed over IPC: the worker-sidecar
    attaches a ``PreparedTokens`` bundle, Python validates the
    ``tokenizer_id`` hash and, on match, assembles ``PreparedBatch``
    directly without calling the HF tokenizer. On any mismatch, drift,
    or unsupported shape the method MUST return ``None`` so the caller
    falls back to ``prepare()`` — correctness always wins over speed.
    """

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.max_sequence_length = 512
        return config

    @staticmethod
    def _make_preprocessor(tokenizer_canonical: bytes) -> tuple[TextPreprocessor, str]:
        """Build a preprocessor whose ``backend_tokenizer.to_str()``
        returns ``tokenizer_canonical``. Returns the preprocessor and
        its expected tokenizer_id (BLAKE3, truncated to 32 hex).
        """
        import blake3

        backend = MagicMock()
        backend.to_str.return_value = tokenizer_canonical.decode("utf-8")
        tokenizer = MagicMock()
        tokenizer.backend_tokenizer = backend
        preprocessor = TextPreprocessor(tokenizer, "fast-model")
        expected = blake3.blake3(tokenizer_canonical).hexdigest()[:32]
        return preprocessor, expected

    def test_tokenizer_id_is_cached_and_matches_blake3(self):
        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, expected = self._make_preprocessor(canonical)

        first = preprocessor.tokenizer_id
        second = preprocessor.tokenizer_id

        assert first == expected
        assert first == second
        # ``backend.to_str`` must be called exactly once regardless of
        # how many callers read the property — it's on the hot path.
        assert preprocessor._tokenizer.backend_tokenizer.to_str.call_count == 1

    def test_tokenizer_id_is_none_for_slow_tokenizer(self):
        """Slow tokenizers don't expose ``backend_tokenizer`` — no hash."""
        tokenizer = MagicMock(spec=[])  # nothing but what we set explicitly
        preprocessor = TextPreprocessor(tokenizer, "slow-model")
        assert preprocessor.tokenizer_id is None

    def test_fast_path_builds_prepared_batch_on_hash_match(self, mock_config):
        from sie_server.ipc_types import PreparedTokens

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        items = [Item(text="foo"), Item(text="bar"), Item(text="baz")]
        prepared_tokens = [
            PreparedTokens(
                input_ids=[[101, 2003, 102]],
                tokenizer_id=tok_id,
                # Omit attention_mask — Rust elides when all-ones. The
                # consumer must rebuild it as [1]*len(input_ids).
                attention_mask=[],
                token_type_ids=[],
                max_seq_len=128,
            ),
            PreparedTokens(
                input_ids=[[101, 3231, 102]],
                tokenizer_id=tok_id,
                attention_mask=[[1, 1, 1]],
                token_type_ids=[],
                max_seq_len=128,
            ),
            PreparedTokens(
                input_ids=[[101, 4, 5, 102]],
                tokenizer_id=tok_id,
                attention_mask=[[1, 1, 1, 1]],
                # All-zero token_type_ids are allowed (BERT-style
                # single-segment default).
                token_type_ids=[[0, 0, 0, 0]],
                max_seq_len=128,
            ),
        ]

        batch = preprocessor.try_prepare_from_prepared_tokens(items, prepared_tokens, config=mock_config)

        assert batch is not None
        assert batch.modality == "text"
        assert batch.size == 3
        assert batch.total_cost == 3 + 3 + 4

        # First item used the elided-attention-mask path — rebuilt as all ones.
        assert batch.items[0].payload.input_ids == [101, 2003, 102]
        assert batch.items[0].payload.attention_mask == [1, 1, 1]
        assert batch.items[0].cost == 3
        assert batch.items[0].original_index == 0

        # Second item had an explicit attention mask; preserved verbatim.
        assert batch.items[1].payload.attention_mask == [1, 1, 1]
        assert batch.items[2].payload.attention_mask == [1, 1, 1, 1]

    def test_fast_path_rejects_tokenizer_id_mismatch(self, mock_config):
        from sie_server.ipc_types import PreparedTokens

        preprocessor, _ = self._make_preprocessor(b'{"version":"1.0","model":{"type":"WordLevel"}}')

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="x")],
            [
                PreparedTokens(
                    input_ids=[[1, 2, 3]],
                    tokenizer_id="a" * 32,  # deliberately wrong
                    attention_mask=[[1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                )
            ],
            config=mock_config,
        )
        assert batch is None

    def test_fast_path_rejects_malformed_attention_mask_shape(self, mock_config):
        from sie_server.ipc_types import PreparedTokens

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="x")],
            [
                PreparedTokens(
                    input_ids=[[1, 2, 3]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                )
            ],
            config=mock_config,
        )
        assert batch is None

    def test_fast_path_rejects_non_zero_token_type_ids(self, mock_config):
        from sie_server.ipc_types import PreparedTokens

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="x")],
            [
                PreparedTokens(
                    input_ids=[[1, 2, 3]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1]],
                    # Segment ids for a sentence-pair encoder are not
                    # part of the Rust-side text payload contract.
                    token_type_ids=[[0, 1, 1]],
                    max_seq_len=128,
                )
            ],
            config=mock_config,
        )
        assert batch is None

    def test_fast_path_hybrid_tokenises_slow_subset_in_python(self, mock_config):
        """Per-item hybrid: items with ``pt=None`` fall back to the
        Python tokeniser inline, other items take the Rust bytes
        unchanged. A single merged ``PreparedBatch`` comes out with
        ``original_index`` preserved so the adapter's output routing
        continues to work.
        """
        from sie_server.ipc_types import PreparedTokens

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        # Wire in a tokeniser stub for the Python fall-back side —
        # only the "b" item (index 1) should flow through.
        preprocessor._tokenizer.return_value = {
            "input_ids": [[101, 999, 102]],
            "attention_mask": [[1, 1, 1]],
        }

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="a"), Item(text="b"), Item(text="c")],
            [
                PreparedTokens(
                    input_ids=[[101, 1, 102]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                ),
                None,  # Rust skipped this item (e.g. empty-text gate).
                PreparedTokens(
                    input_ids=[[101, 2, 3, 102]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                ),
            ],
            config=mock_config,
        )
        assert batch is not None
        assert batch.size == 3

        # Fast-path item (index 0) — Rust bytes verbatim.
        assert batch.items[0].payload.input_ids == [101, 1, 102]
        assert batch.items[0].original_index == 0

        # Slow-path item (index 1) — tokenised in Python via the stub.
        assert batch.items[1].payload.input_ids == [101, 999, 102]
        assert batch.items[1].original_index == 1

        # Fast-path item (index 2) — Rust bytes verbatim.
        assert batch.items[2].payload.input_ids == [101, 2, 3, 102]
        assert batch.items[2].original_index == 2

        # The Python tokeniser was called exactly once, on the
        # one-item slow subset. Multi-call would indicate we re-ran
        # tokenisation on already-prepared inputs.
        assert preprocessor._tokenizer.call_count == 1
        (_, kwargs) = preprocessor._tokenizer.call_args
        assert kwargs["truncation"] is True
        assert kwargs["padding"] is False

    def test_fast_path_returns_none_when_all_items_miss(self, mock_config):
        """All-slow batch: skip the hybrid path entirely so the
        caller's normal ``prepare()`` can run without an extra
        tokeniser call from inside the fast-path helper.
        """
        from sie_server.ipc_types import PreparedTokens as _PT  # noqa: F401, N814

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, _ = self._make_preprocessor(canonical)

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="a"), Item(text="b")],
            [None, None],
            config=mock_config,
        )
        assert batch is None
        # Python tokeniser was NOT called from the fast-path helper —
        # the caller will invoke ``prepare()`` on the full batch.
        assert preprocessor._tokenizer.call_count == 0

    def test_fast_path_hybrid_for_nonzero_token_type_ids(self, mock_config):
        """Non-zero ``token_type_ids`` on a single item falls back to
        Python for that item without killing the whole batch.
        """
        from sie_server.ipc_types import PreparedTokens

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        preprocessor._tokenizer.return_value = {
            "input_ids": [[101, 500, 102]],
            "attention_mask": [[1, 1, 1]],
        }

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="clean"), Item(text="pair")],
            [
                PreparedTokens(
                    input_ids=[[101, 1, 102]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                ),
                PreparedTokens(
                    input_ids=[[101, 2, 3, 102]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1, 1]],
                    token_type_ids=[[0, 1, 1, 0]],  # segment ids.
                    max_seq_len=128,
                ),
            ],
            config=mock_config,
        )
        assert batch is not None
        assert batch.items[0].payload.input_ids == [101, 1, 102]
        # Item 1 went through Python.
        assert batch.items[1].payload.input_ids == [101, 500, 102]
        assert preprocessor._tokenizer.call_count == 1

    def test_fast_path_rejects_whole_batch_on_tokenizer_drift(self, mock_config):
        """Tokenizer_id drift on ANY item collapses the whole batch —
        the hybrid split doesn't help if Rust and Python disagree on
        what tokeniser they're running.
        """
        from sie_server.ipc_types import PreparedTokens

        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="a"), Item(text="b")],
            [
                PreparedTokens(
                    input_ids=[[1, 2, 3]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                ),
                PreparedTokens(
                    input_ids=[[4, 5, 6]],
                    tokenizer_id="deadbeef" * 4,  # drift signal.
                    attention_mask=[[1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=128,
                ),
            ],
            config=mock_config,
        )
        assert batch is None

    def test_fast_path_rejects_when_truncated_above_model_cap(self, mock_config):
        from sie_server.ipc_types import PreparedTokens

        # Model caps at 128; Rust used 512. Staying on Python path
        # ensures the truncation policy the operator configured is
        # the one that takes effect.
        mock_config.max_sequence_length = 128
        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, tok_id = self._make_preprocessor(canonical)

        batch = preprocessor.try_prepare_from_prepared_tokens(
            [Item(text="x")],
            [
                PreparedTokens(
                    input_ids=[[1, 2, 3]],
                    tokenizer_id=tok_id,
                    attention_mask=[[1, 1, 1]],
                    token_type_ids=[],
                    max_seq_len=512,
                )
            ],
            config=mock_config,
        )
        assert batch is None

    def test_fast_path_empty_items_returns_empty_batch(self, mock_config):
        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        preprocessor, _ = self._make_preprocessor(canonical)

        batch = preprocessor.try_prepare_from_prepared_tokens([], [], config=mock_config)
        assert batch is not None
        assert batch.size == 0
        assert batch.total_cost == 0


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    @pytest.fixture
    def mock_processor(self):
        """Create mock image processor."""
        processor = MagicMock()
        # Return processed pixel values
        processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        return processor

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        config = MagicMock()
        return config

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_modality(self, mock_processor):
        """Modality is 'image'."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        assert preprocessor.modality == "image"

    def test_prepare_single_image(self, mock_processor, mock_config, sample_image_bytes):
        """Prepare single image item."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        items = [Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")])]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.modality == "image"
        assert batch.total_cost == 1  # 1 image

        item = batch.items[0]
        assert item.cost == 1
        assert item.original_index == 0
        assert isinstance(item.payload, ImagePayload)
        assert item.payload.original_size == (640, 480)

    def test_prepare_multiple_images(self, mock_processor, mock_config, sample_image_bytes):
        """Prepare multiple image items."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        items: list[Item] = [
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 3
        assert batch.total_cost == 3

        # Check original indices
        indices = [item.original_index for item in batch.items]
        assert indices == [0, 1, 2]

    def test_prepare_text_items_pass_through_aligned(self, mock_processor, mock_config, sample_image_bytes):
        """Text-only items yield empty-payload entries so the batch stays 1:1.

        Dropping them starved the worker's per-item completion check and a
        mixed text+image request hung forever (#3136).
        """
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        items: list[Item] = [
            Item(text="text only"),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(text="also text only"),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 3
        assert batch.total_cost == 3
        assert [item.original_index for item in batch.items] == [0, 1, 2]
        assert batch.items[0].payload.pixel_values is None
        assert batch.items[1].payload.pixel_values is not None
        assert batch.items[2].payload.pixel_values is None

    def test_collate_skips_text_passthrough_entries(self, mock_processor, mock_config, sample_image_bytes):
        """collate() stacks only image payloads from a mixed prepared batch (#3136)."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        items: list[Item] = [
            Item(text="text only"),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
        ]
        batch = preprocessor.prepare(items, config=mock_config)

        result = preprocessor.collate(batch.items, device="cpu")

        assert result["pixel_values"].shape[0] == 1

    def test_collate_all_text_passthrough_returns_empty_tensor(self, mock_processor, mock_config):
        """A prepared batch of only passthrough entries collates to an empty tensor."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        batch = preprocessor.prepare([Item(text="text only")], config=mock_config)

        result = preprocessor.collate(batch.items, device="cpu")

        assert result["pixel_values"].numel() == 0

    def test_prepare_rejects_str_image_data(self, mock_processor, mock_config):
        """Non-bytes image data raises a structured error (defense-in-depth, #1026).

        An un-decoded base64 str on the queue path (where typed msgspec decoding
        doesn't run) must raise InvalidMediaError, not a raw TypeError from
        ``io.BytesIO(str)``.
        """
        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        # msgspec Structs don't validate on direct construction, so this mirrors
        # how the worker builds an Item from an undecoded wire dict.
        items = [Item(images=[{"data": "aGVsbG8=", "format": "png"}])]

        with pytest.raises(InvalidMediaError, match="image data must be bytes, got str"):
            preprocessor.prepare(items, config=mock_config)

    def test_prepare_rejects_undecodable_image_bytes_naming_the_item(self, mock_processor, mock_config):
        """Valid base64-decoded non-image bytes raise a typed error naming the item.

        PIL's UnidentifiedImageError is an OSError, which missed the
        ValueError -> 400 INVALID_INPUT mapping and surfaced as a 500
        INFERENCE_ERROR leaking a BytesIO repr. decode_image maps it to
        InvalidMediaError with the msgspec-style JSON path.
        """
        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        items = [
            Item(text="fine"),
            Item(images=[ImageInput(data=b"valid base64, not an image", format="png")]),
        ]

        with pytest.raises(
            InvalidMediaError,
            match=r"image data is not a decodable image - at `\$\.items\[1\]\.images\[0\]\.data`",
        ):
            preprocessor.prepare(items, config=mock_config)

    def test_prepare_rgba_conversion(self, mock_processor, mock_config):
        """RGBA images are converted to RGB."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        rgba_bytes = buffer.getvalue()

        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        items = [Item(images=[ImageInput(data=rgba_bytes, format="png")])]

        # Should not raise - RGBA converted to RGB
        batch = preprocessor.prepare(items, config=mock_config)
        assert batch.size == 1

    def test_collate_single_image(self, mock_processor):
        """Collate single image item."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        pixel_values = torch.randn(3, 224, 224)
        prepared = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=pixel_values,
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=0,
            )
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert "pixel_values" in result
        assert result["pixel_values"].shape == (1, 3, 224, 224)

    def test_collate_multiple_images(self, mock_processor):
        """Collate batches images correctly."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        prepared = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=torch.randn(3, 224, 224),
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=i,
            )
            for i in range(4)
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert result["pixel_values"].shape == (4, 3, 224, 224)

    def test_collate_empty(self, mock_processor):
        """Collate empty list returns empty tensor."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        result = preprocessor.collate([], device="cpu")

        assert result["pixel_values"].numel() == 0


class TestOpenCLIPImagePreprocessor:
    """Tests for OpenCLIPImagePreprocessor.

    Uses a lambda ``val_preproc`` so the tests are pure unit tests with no
    model download.
    """

    @pytest.fixture
    def fake_val_preproc(self):
        """A torchvision-compose-shaped callable: PIL.Image -> Tensor[3, 224, 224]."""

        def _fn(img: Image.Image) -> torch.Tensor:
            # Deterministic encoding of (width, height) into the first two
            # channels of the corner pixel so tests can verify which image
            # produced the tensor.
            tensor = torch.zeros(3, 224, 224)
            tensor[0, 0, 0] = float(img.width)
            tensor[1, 0, 0] = float(img.height)
            return tensor

        return _fn

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        return MagicMock()

    @pytest.fixture
    def sample_image_bytes(self):
        """Create a 640x480 JPEG."""
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_is_preprocessor(self, fake_val_preproc):
        """OpenCLIPImagePreprocessor satisfies the Preprocessor protocol."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        assert isinstance(preprocessor, Preprocessor)

    def test_modality(self, fake_val_preproc):
        """Modality is 'image'."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        assert preprocessor.modality == "image"

    def test_prepare_runs_val_preproc_and_records_payload(self, fake_val_preproc, mock_config, sample_image_bytes):
        """prepare() invokes val_preproc and stores the resulting tensor on the payload."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items = [Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")])]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.modality == "image"
        assert batch.total_cost == 1

        item = batch.items[0]
        assert item.cost == 1
        assert item.original_index == 0
        assert isinstance(item.payload, ImagePayload)
        assert item.payload.original_size == (640, 480)
        # Tensor shape is [C, H, W] (no batch dim) — same contract as ImagePreprocessor
        assert item.payload.pixel_values.shape == (3, 224, 224)
        # Verify our fake val_preproc actually ran (encoded width/height)
        assert item.payload.pixel_values[0, 0, 0].item() == 640.0
        assert item.payload.pixel_values[1, 0, 0].item() == 480.0

    def test_prepare_rejects_undecodable_image_bytes_naming_the_item(self, fake_val_preproc, mock_config):
        """Same typed-400 contract as ImagePreprocessor for the open_clip path."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items = [Item(images=[ImageInput(data=b"valid base64, not an image", format="png")])]

        with pytest.raises(
            InvalidMediaError,
            match=r"image data is not a decodable image - at `\$\.items\[0\]\.images\[0\]\.data`",
        ):
            preprocessor.prepare(items, config=mock_config)

    def test_prepare_text_items_pass_through_aligned(self, fake_val_preproc, mock_config, sample_image_bytes):
        """Text-only items yield empty-payload entries so the batch stays 1:1 (#3136)."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items: list[Item] = [
            Item(text="text only"),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(text="also text only"),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 3
        assert batch.total_cost == 3
        assert [item.original_index for item in batch.items] == [0, 1, 2]
        assert batch.items[0].payload.pixel_values is None
        assert batch.items[1].payload.pixel_values is not None
        assert batch.items[2].payload.pixel_values is None

    def test_collate_skips_text_passthrough_entries(self, fake_val_preproc, mock_config, sample_image_bytes):
        """collate() stacks only image payloads from a mixed prepared batch (#3136)."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items: list[Item] = [
            Item(text="text only"),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
        ]
        batch = preprocessor.prepare(items, config=mock_config)

        result = preprocessor.collate(batch.items, device="cpu")

        assert result["pixel_values"].shape[0] == 1

    def test_prepare_rgba_conversion(self, fake_val_preproc, mock_config):
        """RGBA images are converted to RGB before val_preproc."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        rgba_bytes = buffer.getvalue()

        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items = [Item(images=[ImageInput(data=rgba_bytes, format="png")])]

        # Should not raise — RGBA → RGB happens before the lambda is invoked.
        batch = preprocessor.prepare(items, config=mock_config)
        assert batch.size == 1
        assert batch.items[0].payload.pixel_values.shape == (3, 224, 224)

    def test_collate_stacks_into_batch(self, fake_val_preproc):
        """collate() stacks [C, H, W] payloads into [B, C, H, W] on the requested device."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        prepared = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=torch.randn(3, 224, 224),
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=i,
            )
            for i in range(4)
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert "pixel_values" in result
        assert result["pixel_values"].shape == (4, 3, 224, 224)

    def test_collate_empty_returns_empty_tensor(self, fake_val_preproc):
        """collate() on an empty list returns a 0-element tensor (matches ImagePreprocessor)."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        result = preprocessor.collate([], device="cpu")

        assert result["pixel_values"].numel() == 0


class TestNemoColEmbedPreprocessor:
    """Tests for NemoColEmbedPreprocessor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_model_config(self):
        """Create mock model config."""
        config = MagicMock()
        config.template = "bidirectional-llama-retriever"
        return config

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.fixture
    def mock_config(self):
        """Create mock model config for prepare()."""
        config = MagicMock()
        config.max_sequence_length = 10240
        return config

    def test_modality(self, mock_tokenizer, mock_model_config):
        """NemoColEmbedPreprocessor reports correct modality."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        assert preprocessor.modality == "image"

    def test_build_prompt_single_tile(self, mock_tokenizer, mock_model_config):
        """Build prompt creates correct format for single tile."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
            num_image_token=256,
        )

        prompt = preprocessor._build_prompt(num_tiles=1)

        # Should have 256 IMG_CONTEXT tokens for 1 tile
        assert prompt.startswith("passage: <img>")
        assert prompt.endswith("</img> ")
        assert prompt.count("<IMG_CONTEXT>") == 256

    def test_build_prompt_multiple_tiles(self, mock_tokenizer, mock_model_config):
        """Build prompt creates correct format for multiple tiles."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
            num_image_token=256,
        )

        prompt = preprocessor._build_prompt(num_tiles=3)

        # Should have 256 * 3 = 768 IMG_CONTEXT tokens for 3 tiles
        assert prompt.count("<IMG_CONTEXT>") == 768

    def test_dynamic_preprocess_single_tile(self):
        """Dynamic preprocess creates single tile for square image."""
        from sie_server.core.preprocessor import _dynamic_preprocess

        img = Image.new("RGB", (448, 448), color="blue")
        tiles = _dynamic_preprocess(img, image_size=448, max_num=6)

        # Square 448x448 image should produce 1 tile
        assert len(tiles) == 1
        assert tiles[0].size == (448, 448)

    def test_dynamic_preprocess_multiple_tiles(self):
        """Dynamic preprocess creates multiple tiles for wide image."""
        from sie_server.core.preprocessor import _dynamic_preprocess

        # Wide image should create multiple tiles
        img = Image.new("RGB", (896, 448), color="green")
        tiles = _dynamic_preprocess(img, image_size=448, max_num=6)

        # 2:1 aspect ratio should produce 2 tiles
        assert len(tiles) == 2

    def test_prepare_creates_payload(self, mock_tokenizer, mock_model_config, mock_config, sample_image_bytes):
        """Prepare creates NemoColEmbedPayload with correct fields."""
        from sie_server.core.prepared import NemoColEmbedPayload
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        # Mock tokenizer return value
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        items: list[Item] = [
            Item(images=[ImageInput(data=sample_image_bytes)]),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.modality == "image"
        assert len(batch.items) == 1
        assert batch.total_cost > 0

        payload = batch.items[0].payload
        assert isinstance(payload, NemoColEmbedPayload)
        assert payload.pixel_values is not None
        assert payload.input_ids is not None
        assert payload.attention_mask is not None
        assert payload.num_tiles > 0
        assert payload.original_size == (640, 480)

    def test_collate_concatenates_variable_tiles(self, mock_tokenizer, mock_model_config):
        """Collate concatenates pixel values from items with different tile counts."""
        from sie_server.core.prepared import NemoColEmbedPayload, PreparedItem
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        mock_tokenizer.pad_token_id = 0

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        # Create items with different tile counts
        prepared = [
            PreparedItem(
                payload=NemoColEmbedPayload(
                    pixel_values=torch.randn(2, 3, 448, 448),  # 2 tiles
                    input_ids=torch.tensor([1, 2, 3]),
                    attention_mask=torch.tensor([1, 1, 1]),
                    num_tiles=2,
                    original_size=(896, 448),
                ),
                cost=2,
                original_index=0,
            ),
            PreparedItem(
                payload=NemoColEmbedPayload(
                    pixel_values=torch.randn(1, 3, 448, 448),  # 1 tile
                    input_ids=torch.tensor([1, 2]),
                    attention_mask=torch.tensor([1, 1]),
                    num_tiles=1,
                    original_size=(448, 448),
                ),
                cost=1,
                original_index=1,
            ),
        ]

        result = preprocessor.collate(prepared, device="cpu")

        # pixel_values should be concatenated: 2 + 1 = 3 tiles total
        assert result["pixel_values"].shape == (3, 3, 448, 448)

        # input_ids should be padded to max length (3)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_collate_empty(self, mock_tokenizer, mock_model_config):
        """Collate empty list returns empty tensors."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        result = preprocessor.collate([], device="cpu")

        assert result["pixel_values"].numel() == 0
        assert result["input_ids"].numel() == 0
        assert result["attention_mask"].numel() == 0


class TestDetectionPreprocessorMaxSide:
    """An oversized photo is shrunk before the image processor, and boxes still map to the pixels sent."""

    @staticmethod
    def _jpeg(width: int, height: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (width, height), (120, 40, 200)).save(buf, "JPEG")
        return buf.getvalue()

    def _prepare(self, width: int, height: int, max_side: int | None) -> tuple[tuple[int, int], tuple[int, int]]:
        seen: dict[str, tuple[int, int]] = {}

        def fake_processor(*, images, return_tensors):
            seen["size"] = images.size
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}

        pre = DetectionPreprocessor(image_processor=fake_processor, model_name="m", max_side=max_side)
        item = Item(images=[{"data": self._jpeg(width, height), "format": "jpeg"}])
        batch = pre.prepare([item], config=MagicMock())
        return seen["size"], batch.items[0].payload.original_size

    def test_large_photo_is_shrunk_but_keeps_its_original_size(self) -> None:
        processed, original = self._prepare(8192, 1728, 1920)
        assert processed == (1920, 405)
        assert original == (8192, 1728)

    def test_photo_within_the_cap_is_untouched(self) -> None:
        processed, original = self._prepare(1920, 1080, 1920)
        assert processed == (1920, 1080)
        assert original == (1920, 1080)

    def test_no_cap_keeps_the_old_behaviour(self) -> None:
        processed, _ = self._prepare(4000, 3000, None)
        assert processed == (4000, 3000)
