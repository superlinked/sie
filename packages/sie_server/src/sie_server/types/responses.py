"""Response types for SIE Server API.

These types define the structure of API responses for encode, score, and extract endpoints.
Using TypedDict for zero runtime overhead.
"""

from enum import StrEnum
from typing import Any, TypedDict

from sie_server.types.outputs import EncodeResult


class TimingInfo(TypedDict, total=False):
    """Server-side timing breakdown for a request.

    Attributes:
        total_ms: Total end-to-end time.
        queue_ms: Time waiting in queue.
        tokenization_ms: Time tokenizing input.
        inference_ms: GPU forward pass time.
        postprocessing_ms: Postprocessor transforms time.
    """

    total_ms: float
    queue_ms: float
    tokenization_ms: float
    inference_ms: float
    postprocessing_ms: float | None


class EncodeResponse(TypedDict, total=False):
    """Response body for POST /v1/encode/{model}.

    Attributes:
        model: Model name used for encoding.
        items: Encoded results, one per input item.
        timing: Server-side timing breakdown.
    """

    model: str
    items: list[EncodeResult]
    timing: TimingInfo | None


class ScoreEntry(TypedDict):
    """A single score result from reranking.

    Attributes:
        item_id: Item ID (echoed from request).
        score: Relevance score (higher = more relevant).
        rank: Rank position (0 = most relevant).
    """

    item_id: str | None
    score: float
    rank: int


class ScoreResponse(TypedDict, total=False):
    """Response body for POST /v1/score/{model}.

    Attributes:
        model: Model name used for scoring.
        query_id: Query ID (echoed from request).
        scores: Scores sorted by relevance (descending).
    """

    model: str
    query_id: str | None
    scores: list[ScoreEntry]


class Entity(TypedDict, total=False):
    """A single extracted entity (NER span or document region).

    Attributes:
        text: Extracted text span.
        label: Entity type label.
        score: Confidence score.
        start: Start character offset (None for image-based).
        end: End character offset (None for image-based).
        bbox: Bounding box [x, y, w, h] in pixels.
    """

    text: str
    label: str
    score: float
    start: int | None
    end: int | None
    bbox: list[int] | None


class Relation(TypedDict):
    """A single extracted relation triple.

    Attributes:
        head: Head entity text.
        tail: Tail entity text.
        relation: Relation type label.
        score: Confidence score.
    """

    head: str
    tail: str
    relation: str
    score: float


class Classification(TypedDict):
    """A single classification result.

    Attributes:
        label: Classification label.
        score: Confidence score.
    """

    label: str
    score: float


class DetectedObject(TypedDict):
    """A single detected object with bounding box.

    Attributes:
        label: Object class label.
        score: Confidence score.
        bbox: Bounding box [x, y, w, h] in pixels.
    """

    label: str
    score: float
    bbox: list[int]


class ExtractResult(TypedDict, total=False):
    """Result of extracting from a single item.

    Attributes:
        id: Item ID (echoed from request).
        entities: NER entities.
        relations: Relation triples.
        classifications: Classifications.
        objects: Detected objects.
        data: Structured extraction results.
    """

    id: str
    entities: list[Entity]
    relations: list[Relation]
    classifications: list[Classification]
    objects: list[DetectedObject]
    data: dict[str, Any]


# Backwards compatibility alias
EntityResult = Entity


class ExtractResponse(TypedDict, total=False):
    """Response body for POST /v1/extract/{model}.

    Attributes:
        model: Model name used for extraction.
        items: Extraction results, one per input item.
    """

    model: str
    items: list[ExtractResult]


class ErrorCode(StrEnum):
    """Standard error codes for API errors."""

    INVALID_INPUT = "INVALID_INPUT"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    LORA_LOADING = "LORA_LOADING"  # LoRA adapter loading in progress - SDK should retry
    MODEL_LOADING = "MODEL_LOADING"  # Model loading in progress - SDK should retry
    INFERENCE_ERROR = "INFERENCE_ERROR"
    QUEUE_FULL = "QUEUE_FULL"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"  # GPU OOM


class ErrorDetail(TypedDict, total=False):
    """Detailed error information.

    Attributes:
        code: Error code.
        message: Human-readable error message.
        request_id: Request ID for debugging.
    """

    code: str  # ErrorCode.value
    message: str
    request_id: str | None


class ErrorResponse(TypedDict):
    """Error response body for all endpoints.

    Attributes:
        error: Error details.
    """

    error: ErrorDetail
