"""Tests for NATS JetStream queue types and helpers."""

from sie_sdk.queue_types import (
    INLINE_THRESHOLD_BYTES,
    denormalize_model_id,
    normalize_model_id,
    work_consumer_name,
    work_stream_name,
    work_subject,
)


class TestNormalizeModelId:
    def test_slash_replaced(self) -> None:
        assert normalize_model_id("BAAI/bge-m3") == "BAAI__bge-m3"

    def test_no_slash(self) -> None:
        assert normalize_model_id("bge-m3") == "bge-m3"

    def test_multiple_slashes(self) -> None:
        assert normalize_model_id("org/group/model") == "org__group__model"


class TestWorkSubject:
    def test_default_pool(self) -> None:
        assert work_subject("BAAI/bge-m3", "_default") == "sie.work.BAAI__bge-m3._default"

    def test_named_pool(self) -> None:
        assert work_subject("BAAI/bge-m3", "eval-pool") == "sie.work.BAAI__bge-m3.eval-pool"


class TestWorkStreamName:
    def test_format(self) -> None:
        assert work_stream_name("BAAI/bge-m3") == "WORK_BAAI__bge-m3"


class TestWorkConsumerName:
    def test_format(self) -> None:
        assert work_consumer_name("default", "_default") == "default__default"

    def test_sglang_pool(self) -> None:
        assert work_consumer_name("sglang", "eval") == "sglang_eval"


class TestDenormalizeModelId:
    def test_double_underscore_to_slash(self) -> None:
        assert denormalize_model_id("BAAI__bge-m3") == "BAAI/bge-m3"

    def test_dot_encoding_reversed(self) -> None:
        assert denormalize_model_id("org__model_dot_v2") == "org/model.v2"

    def test_no_encoding(self) -> None:
        assert denormalize_model_id("bge-m3") == "bge-m3"

    def test_multiple_slashes_roundtrip(self) -> None:
        original = "org/group/model"
        assert denormalize_model_id(normalize_model_id(original)) == original

    def test_roundtrip_simple(self) -> None:
        """denormalize(normalize(id)) == id for typical HuggingFace IDs."""
        for model_id in ["BAAI/bge-m3", "intfloat/e5-base-v2", "sentence-transformers/all-MiniLM-L6-v2"]:
            assert denormalize_model_id(normalize_model_id(model_id)) == model_id

    def test_roundtrip_with_dots(self) -> None:
        """Roundtrip works for IDs containing dots."""
        original = "org/model.v2.1"
        assert denormalize_model_id(normalize_model_id(original)) == original

    def test_collision_with_literal_double_underscore(self) -> None:
        """IDs with literal __ are NOT roundtrippable — document the edge case."""
        original = "org/a__b"
        normalized = normalize_model_id(original)
        # normalize: "org/a__b" → "org__a__b" → denormalize: "org/a/b" (wrong!)
        assert denormalize_model_id(normalized) != original
        assert denormalize_model_id(normalized) == "org/a/b"


class TestInlineThreshold:
    def test_is_1mb(self) -> None:
        assert INLINE_THRESHOLD_BYTES == 1_048_576
