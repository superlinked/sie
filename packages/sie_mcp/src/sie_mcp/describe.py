"""The ``describe_image`` image-understanding job.

Stateless orchestration over two existing SIE primitives, run entirely
client-side in the MCP edge so the gateway and workers stay stateless per
request:

- **Caption** — the image's bytes go to the Florence-2 adapter via the
  ``extract`` path with a ``<CAPTION>`` / ``<DETAILED_CAPTION>`` task; the
  caption comes back as the extracted entity text.
- **Tags (zero-shot classify)** — done here in the edge: candidate labels and
  the image are embedded with ``encode`` (SigLIP/CLIP) and the top-k labels by
  cosine similarity to the image embedding are returned. No server-side
  endpoint — it is orchestration over the existing encode primitive.
"""

import asyncio
import base64
import binascii
import math
from typing import Any, Protocol, TypedDict

from sie_mcp.config import DEFAULT_MAX_IMAGE_BYTES

# Florence-2 caption task tokens; the detailed variant yields a longer caption.
_CAPTION_TASK = "<CAPTION>"
_DETAILED_CAPTION_TASK = "<DETAILED_CAPTION>"


class DescribeImageError(Exception):
    """Raised when the cluster could not caption or classify the image."""


class DescribeClient(Protocol):
    """The slice of ``SIEAsyncClient`` this job needs (keeps the job unit-testable)."""

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        """Run an extract request against the SIE cluster."""

    async def encode(self, model: str, items: Any, **kwargs: Any) -> Any:
        """Run an encode request against the SIE cluster."""


class Tag(TypedDict):
    label: str
    score: float


class DescribeImageResult(TypedDict):
    caption: str
    tags: list[Tag]


def _decode_base64(image_base64: str, *, max_bytes: int) -> bytes:
    # Reject oversize payloads from the base64 length before allocating the decoded
    # buffer (decoded size ≈ 3/4 of the encoded length), then enforce the exact bound.
    if (len(image_base64) // 4) * 3 > max_bytes:
        msg = f"image exceeds the {max_bytes}-byte limit"
        raise DescribeImageError(msg)
    try:
        data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = f"image_base64 is not valid base64: {exc}"
        raise DescribeImageError(msg) from exc
    if len(data) > max_bytes:
        msg = f"image exceeds the {max_bytes}-byte limit"
        raise DescribeImageError(msg)
    return data


def _caption_text(result: Any) -> str:
    """Pull the caption string out of a Florence-2 ExtractResult.

    Captioning returns a single entity whose ``text`` is the caption; take the
    first entity that carries text. Returns ``""`` when none is present.
    """
    entities = result.get("entities") if isinstance(result, dict) else None
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict) and entity.get("text"):
                return str(entity["text"])
    return ""


def _dense_vector(result: Any) -> list[float]:
    """Extract the dense embedding from an EncodeResult as ``list[float]``."""
    dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
    if dense is None:
        msg = "encode result missing dense embedding"
        raise DescribeImageError(msg)
    # The real SDK returns a numpy array (``.tolist()``); the list branch is for
    # test fakes that pass a plain list. Keep both — do not "simplify" either away.
    return dense.tolist() if hasattr(dense, "tolist") else [float(x) for x in dense]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; SigLIP/CLIP embeddings are not normalized by default.

    Callers must pass equal-length vectors (``_top_k_tags`` validates this); strict
    zip turns any dimension mismatch into an error rather than a silent truncation.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _top_k_tags(
    image_vec: list[float],
    label_vecs: list[list[float]],
    labels: list[str],
    top_k: int,
) -> list[Tag]:
    """Rank labels by cosine similarity to the image, keep the top-k (argmax).

    Validates the embedding shapes up front and raises ``DescribeImageError`` on any
    mismatch, rather than silently dropping labels via a lenient zip.
    """
    if not image_vec:
        msg = "image embedding is empty"
        raise DescribeImageError(msg)
    if len(labels) != len(label_vecs):
        msg = f"got {len(label_vecs)} label embeddings for {len(labels)} labels"
        raise DescribeImageError(msg)
    scored: list[Tag] = []
    for label, label_vec in zip(labels, label_vecs, strict=True):
        if len(label_vec) != len(image_vec):
            msg = f"label embedding dim {len(label_vec)} != image embedding dim {len(image_vec)}"
            raise DescribeImageError(msg)
        scored.append(Tag(label=label, score=_cosine(image_vec, label_vec)))
    scored.sort(key=lambda tag: tag["score"], reverse=True)
    return scored[: max(top_k, 0)]


async def describe_image(
    client: DescribeClient,
    *,
    image_base64: str,
    labels: list[str],
    caption_model: str,
    embed_model: str,
    detailed: bool = False,
    top_k: int = 5,
    gpu: str | None = None,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> DescribeImageResult:
    """Caption an image (Florence-2) and tag it via zero-shot classify (SigLIP/CLIP).

    The image's bytes go to the Florence-2 ``extract`` path for the caption; the
    same bytes plus the candidate ``labels`` are embedded with ``encode`` and the
    top-k labels by cosine similarity to the image become the tags. The tag step
    is skipped entirely (no encode call) when there are no labels or ``top_k <= 0``
    — a caption-only call costs no embedding work — and ``tags`` is empty. The
    decoded image is rejected before any cluster call once it exceeds
    ``max_image_bytes`` (memory-exhaustion guard, mirrors the document path).
    """
    data = _decode_base64(image_base64, max_bytes=max_image_bytes)
    image_item = {"images": [data]}

    task = _DETAILED_CAPTION_TASK if detailed else _CAPTION_TASK
    caption_result = await client.extract(caption_model, image_item, options={"task": task}, gpu=gpu)
    caption = _caption_text(caption_result)

    tags: list[Tag] = []
    if labels and top_k > 0:
        # ``top_k <= 0`` keeps no tags, so skip the embeds entirely — no point paying
        # for the label/image encodes on a caption-only call.
        # The label-batch and image embeds are independent — run them concurrently
        # so the tag path costs one round-trip of wall-clock, not two.
        label_results, image_result = await asyncio.gather(
            client.encode(embed_model, [{"text": label} for label in labels], gpu=gpu),
            client.encode(embed_model, image_item, gpu=gpu),
        )
        label_vecs = [_dense_vector(result) for result in label_results]
        image_vec = _dense_vector(image_result)
        tags = _top_k_tags(image_vec, label_vecs, labels, top_k)

    return DescribeImageResult(caption=caption, tags=tags)
