from typing import Any

import numpy as np
import pytest
from sie_mcp import describe
from sie_mcp.describe import DescribeImageError
from sie_sdk import RequestError

_CAPTION_MODEL = "microsoft/Florence-2-base-ft"
_EMBED_MODEL = "openai/clip-vit-base-patch32"


class _FakeClient:
    """Records extract()/encode() calls and returns canned result-shaped dicts.

    ``encode`` mirrors the SDK contract: a list of items yields a list of
    EncodeResults (the label embeddings), a single item yields one (the image).
    """

    def __init__(
        self,
        *,
        caption_entities: Any,
        label_vectors: dict[str, Any] | None = None,
        image_vector: Any = None,
    ) -> None:
        # Explicit None checks (not ``or``) so numpy-array vectors don't trip the
        # ambiguous-truth-value error.
        self._caption_entities = caption_entities
        self._label_vectors = label_vectors if label_vectors is not None else {}
        self._image_vector = image_vector if image_vector is not None else []
        self.extract_calls: list[dict[str, Any]] = []
        self.encode_calls: list[dict[str, Any]] = []

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.extract_calls.append({"model": model, "items": items, **kwargs})
        return {"entities": self._caption_entities}

    async def encode(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.encode_calls.append({"model": model, "items": items, **kwargs})
        if isinstance(items, list):
            return [{"dense": self._label_vectors[item["text"]]} for item in items]
        return {"dense": self._image_vector}


def _client(**kwargs: Any) -> _FakeClient:
    return _FakeClient(caption_entities=[{"text": "a cat", "label": "caption", "score": 1.0}], **kwargs)


async def _describe(client: _FakeClient, **kwargs: Any) -> Any:
    params: dict[str, Any] = {
        "image_base64": "",
        "labels": [],
        "caption_model": _CAPTION_MODEL,
        "embed_model": _EMBED_MODEL,
    }
    params.update(kwargs)
    return await describe.describe_image(client, **params)


async def test_returns_caption_and_tags() -> None:
    client = _client(
        label_vectors={"cat": [1.0, 0.0], "dog": [0.0, 1.0], "bird": [0.7, 0.7]},
        image_vector=[1.0, 0.0],
    )

    result = await _describe(client, labels=["cat", "dog", "bird"], top_k=3)

    assert result["caption"] == "a cat"
    assert [tag["label"] for tag in result["tags"]] == ["cat", "bird", "dog"]
    assert result["tags"][0]["score"] == pytest.approx(1.0)
    # Scores are sorted descending.
    scores = [tag["score"] for tag in result["tags"]]
    assert scores == sorted(scores, reverse=True)


async def test_zero_shot_argmax_picks_closest_unnormalized() -> None:
    # Cosine normalizes, so an unnormalized image vector still ranks by direction.
    client = _client(
        label_vectors={"cat": [2.0, 0.0], "dog": [0.0, 3.0]},
        image_vector=[5.0, 0.0],
    )

    result = await _describe(client, labels=["cat", "dog"], top_k=1)

    assert [tag["label"] for tag in result["tags"]] == ["cat"]
    assert result["tags"][0]["score"] == pytest.approx(1.0)


async def test_top_k_limits_tags() -> None:
    client = _client(
        label_vectors={"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0]},
        image_vector=[1.0, 0.0],
    )

    result = await _describe(client, labels=["a", "b", "c"], top_k=2)

    assert len(result["tags"]) == 2
    assert [tag["label"] for tag in result["tags"]] == ["a", "b"]


async def test_top_k_zero_returns_no_tags_and_skips_encode() -> None:
    client = _client(label_vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]}, image_vector=[1.0, 0.0])

    result = await _describe(client, labels=["a", "b"], top_k=0)

    assert result["tags"] == []
    # top_k<=0 keeps no tags, so no embedding round-trip should be made at all.
    assert client.encode_calls == []


async def test_top_k_negative_returns_no_tags_and_skips_encode() -> None:
    client = _client(label_vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]}, image_vector=[1.0, 0.0])

    result = await _describe(client, labels=["a", "b"], top_k=-1)

    assert result["tags"] == []
    assert client.encode_calls == []


async def test_top_k_larger_than_labels_returns_all() -> None:
    client = _client(
        label_vectors={"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.7, 0.7]},
        image_vector=[1.0, 0.0],
    )

    result = await _describe(client, labels=["a", "b", "c"], top_k=99)

    assert [tag["label"] for tag in result["tags"]] == ["a", "c", "b"]


async def test_handles_numpy_dense_vectors() -> None:
    # The real SDK returns numpy arrays for ``dense``; exercise that ``.tolist()`` path.
    client = _client(
        label_vectors={"cat": np.array([1.0, 0.0]), "dog": np.array([0.0, 1.0])},
        image_vector=np.array([2.0, 0.0]),
    )

    result = await _describe(client, labels=["cat", "dog"], top_k=2)

    assert [tag["label"] for tag in result["tags"]] == ["cat", "dog"]
    assert result["tags"][0]["score"] == pytest.approx(1.0)


async def test_default_task_is_plain_caption() -> None:
    client = _client(label_vectors={"cat": [1.0]}, image_vector=[1.0])

    await _describe(client, labels=["cat"])

    assert client.extract_calls[0]["options"] == {"task": "<CAPTION>"}
    assert client.extract_calls[0]["model"] == _CAPTION_MODEL


async def test_detailed_uses_detailed_caption_task() -> None:
    client = _client(label_vectors={"cat": [1.0]}, image_vector=[1.0])

    await _describe(client, labels=["cat"], detailed=True)

    assert client.extract_calls[0]["options"] == {"task": "<DETAILED_CAPTION>"}


async def test_empty_labels_skips_encode() -> None:
    client = _client()

    result = await _describe(client, labels=[])

    assert result["caption"] == "a cat"
    assert result["tags"] == []
    assert client.encode_calls == []


async def test_encode_uses_embed_model_and_image_item() -> None:
    client = _client(label_vectors={"cat": [1.0]}, image_vector=[1.0])

    await _describe(client, labels=["cat"])

    # Two encode calls (run concurrently, so order is not asserted): the label
    # batch (list of text items) and the image (single item).
    assert len(client.encode_calls) == 2
    assert all(call["model"] == _EMBED_MODEL for call in client.encode_calls)
    items = [call["items"] for call in client.encode_calls]
    assert [{"text": "cat"}] in items
    assert any(isinstance(it, dict) and "images" in it for it in items)


async def test_caption_missing_returns_empty_string() -> None:
    client = _FakeClient(caption_entities=[])

    result = await _describe(client, labels=[])

    assert result["caption"] == ""


async def test_rejects_invalid_base64() -> None:
    client = _client()

    with pytest.raises(DescribeImageError):
        await _describe(client, image_base64="!!not base64!!")


async def test_rejects_oversize_image_before_any_cluster_call() -> None:
    client = _client(label_vectors={"cat": [1.0]}, image_vector=[1.0])
    # 400 base64 chars decode to ~300 bytes, well over the 4-byte cap; the
    # pre-decode length check must reject it before any extract/encode round-trip.
    oversize = "A" * 400

    with pytest.raises(DescribeImageError):
        await _describe(client, image_base64=oversize, labels=["cat"], max_image_bytes=4)

    assert client.extract_calls == []
    assert client.encode_calls == []


async def test_raises_on_missing_dense_embedding() -> None:
    client = _client(label_vectors={"cat": [1.0]}, image_vector=[1.0])
    client._image_vector = None  # type: ignore[assignment]  # simulate a malformed encode result

    with pytest.raises(DescribeImageError):
        await _describe(client, labels=["cat"])


async def test_raises_on_embedding_dim_mismatch() -> None:
    # A label vector whose dimension differs from the image vector must raise rather
    # than silently drop the label via a lenient zip.
    client = _client(label_vectors={"cat": [1.0, 0.0]}, image_vector=[1.0])

    with pytest.raises(DescribeImageError):
        await _describe(client, labels=["cat"], top_k=1)


async def test_names_the_setting_when_the_caption_model_is_not_served() -> None:
    class _NoCaptioner(_FakeClient):
        async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
            raise RequestError(f"Model '{model}' not found", code="MODEL_NOT_FOUND", status_code=404)

    client = _NoCaptioner(caption_entities=[])

    with pytest.raises(DescribeImageError, match="SIE_MCP_CAPTION_MODEL"):
        await _describe(client)


async def test_names_the_setting_when_the_embed_model_is_not_served() -> None:
    class _NoEmbedder(_FakeClient):
        async def encode(self, model: str, items: Any, **kwargs: Any) -> Any:
            raise RequestError(f"Model '{model}' not found", code="MODEL_NOT_FOUND", status_code=404)

    client = _NoEmbedder(caption_entities=[{"text": "a cat"}])

    with pytest.raises(DescribeImageError, match="SIE_MCP_EMBED_MODEL"):
        await _describe(client, labels=["cat"], top_k=1)


async def test_other_request_errors_pass_through() -> None:
    class _BadRequest(_FakeClient):
        async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
            raise RequestError("bad request", code="INVALID_INPUT", status_code=400)

    with pytest.raises(RequestError, match="bad request"):
        await _describe(_BadRequest(caption_entities=[]))
