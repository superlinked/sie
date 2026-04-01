"""Tests for prepared item types."""

import torch
from sie_server.core.prepared import (
    AudioPayload,
    ImagePayload,
    MixedPayload,
    PreparedBatch,
    PreparedItem,
    TextPayload,
)


class TestTextPayload:
    """Tests for TextPayload."""

    def test_token_count(self):
        """Token count matches input_ids length."""
        payload = TextPayload(
            input_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
        )
        assert payload.token_count == 5

    def test_empty_payload(self):
        """Empty payload has zero tokens."""
        payload = TextPayload(input_ids=[], attention_mask=[])
        assert payload.token_count == 0


class TestImagePayload:
    """Tests for ImagePayload."""

    def test_creation(self):
        """ImagePayload stores pixel values and size."""
        pixel_values = torch.randn(3, 224, 224)
        payload = ImagePayload(
            pixel_values=pixel_values,
            original_size=(640, 480),
        )
        assert payload.pixel_values.shape == (3, 224, 224)
        assert payload.original_size == (640, 480)


class TestAudioPayload:
    """Tests for AudioPayload."""

    def test_creation(self):
        """AudioPayload stores waveform and metadata."""
        waveform = torch.randn(1, 16000)
        payload = AudioPayload(
            waveform=waveform,
            sample_rate=16000,
            duration_s=1.0,
        )
        assert payload.waveform.shape == (1, 16000)
        assert payload.sample_rate == 16000
        assert payload.duration_s == 1.0


class TestPreparedItem:
    """Tests for PreparedItem."""

    def test_text_item(self):
        """PreparedItem with TextPayload."""
        payload = TextPayload(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])
        item = PreparedItem(payload=payload, cost=3, original_index=0)

        assert item.cost == 3
        assert item.original_index == 0
        assert item.payload.token_count == 3

    def test_image_item(self):
        """PreparedItem with ImagePayload."""
        pixel_values = torch.randn(3, 224, 224)
        payload = ImagePayload(pixel_values=pixel_values, original_size=(640, 480))
        item = PreparedItem(payload=payload, cost=1, original_index=5)

        assert item.cost == 1
        assert item.original_index == 5


class TestPreparedBatch:
    """Tests for PreparedBatch."""

    def test_size(self):
        """Batch size equals number of items."""
        items = [
            PreparedItem(
                payload=TextPayload(input_ids=[1, 2], attention_mask=[1, 1]),
                cost=2,
                original_index=i,
            )
            for i in range(5)
        ]
        batch = PreparedBatch(items=items, total_cost=10, modality="text")

        assert batch.size == 5
        assert batch.total_cost == 10
        assert batch.modality == "text"

    def test_sorted_by_cost(self):
        """sorted_by_cost returns items in ascending cost order."""
        items = [
            PreparedItem(
                payload=TextPayload(input_ids=[1] * cost, attention_mask=[1] * cost),
                cost=cost,
                original_index=i,
            )
            for i, cost in enumerate([5, 2, 8, 1, 3])
        ]
        batch = PreparedBatch(items=items, total_cost=19, modality="text")

        sorted_batch = batch.sorted_by_cost()

        costs = [item.cost for item in sorted_batch.items]
        assert costs == [1, 2, 3, 5, 8]
        # Total cost preserved
        assert sorted_batch.total_cost == 19
        assert sorted_batch.modality == "text"

    def test_empty_batch(self):
        """Empty batch has size 0."""
        batch = PreparedBatch(items=[], total_cost=0, modality="text")
        assert batch.size == 0

    def test_image_batch(self):
        """Batch with image modality."""
        items = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=torch.randn(3, 224, 224),
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=i,
            )
            for i in range(3)
        ]
        batch = PreparedBatch(items=items, total_cost=3, modality="image")

        assert batch.size == 3
        assert batch.modality == "image"


class TestMixedPayload:
    """Tests for MixedPayload."""

    def test_text_only(self):
        """MixedPayload with only text."""
        payload = MixedPayload(text=TextPayload(input_ids=[1, 2, 3], attention_mask=[1, 1, 1]))
        assert payload.modalities == ["text"]

    def test_image_only(self):
        """MixedPayload with only image."""
        payload = MixedPayload(
            image=ImagePayload(
                pixel_values=torch.randn(3, 224, 224),
                original_size=(640, 480),
            )
        )
        assert payload.modalities == ["image"]

    def test_text_and_image(self):
        """MixedPayload with text and image."""
        payload = MixedPayload(
            text=TextPayload(input_ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            image=ImagePayload(
                pixel_values=torch.randn(3, 224, 224),
                original_size=(640, 480),
            ),
        )
        assert payload.modalities == ["text", "image"]

    def test_all_modalities(self):
        """MixedPayload with all modalities."""
        payload = MixedPayload(
            text=TextPayload(input_ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            image=ImagePayload(
                pixel_values=torch.randn(3, 224, 224),
                original_size=(640, 480),
            ),
            audio=AudioPayload(
                waveform=torch.randn(1, 16000),
                sample_rate=16000,
                duration_s=1.0,
            ),
        )
        assert payload.modalities == ["text", "image", "audio"]

    def test_empty_payload(self):
        """MixedPayload with no modalities."""
        payload = MixedPayload()
        assert payload.modalities == []
