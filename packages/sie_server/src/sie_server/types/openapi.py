"""Pydantic models for OpenAPI documentation.

These models mirror the TypedDict types in sie_server.types but are Pydantic BaseModel
classes that FastAPI uses to generate OpenAPI schemas in Swagger UI.

The actual request/response handling uses TypedDict for zero overhead, while these
models provide rich documentation with descriptions and examples.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from sie_server.core.score_cost import MAX_SCORE_ITEMS


# Request models
class ImageInputModel(BaseModel):
    """Image input for multimodal models."""

    data: bytes = Field(..., description="Image data as bytes")
    format: str | None = Field(default=None, description="Image format hint: 'jpeg', 'png', etc.")


class DocumentInputModel(BaseModel):
    """Document input for composite-document extractors (PDF, DOCX, HTML, ...)."""

    data: bytes = Field(..., description="Document bytes (raw file content)")
    format: str | None = Field(default=None, description="Document format hint: 'pdf', 'docx', 'html', etc.")


class ItemModel(BaseModel):
    """A single item to encode."""

    id: str | None = Field(default=None, description="Optional identifier for this item. Returned in response.")
    text: str | None = Field(default=None, description="Text content to encode", examples=["Hello, world!"])
    images: list[ImageInputModel] | None = Field(default=None, description="Images for multimodal models")
    document: DocumentInputModel | None = Field(
        default=None, description="Document for composite-document extractors (PDF, DOCX, HTML, ...)"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Arbitrary metadata. Returned in response.")

    model_config = {"extra": "allow"}


class EncodeParamsModel(BaseModel):
    """Parameters for encode requests."""

    output_types: list[Literal["dense", "sparse", "multivector"]] | None = Field(
        default=None, description="Output types to return"
    )
    instruction: str | None = Field(default=None, description="Task instruction for instruction-tuned models")
    output_dtype: Literal["float32", "float16", "int8", "binary"] | None = Field(
        default=None, description="Output dtype"
    )
    options: dict[str, Any] | None = Field(default=None, description="Runtime options")


class EncodeRequestModel(BaseModel):
    """Request body for encode endpoint."""

    items: list[ItemModel] = Field(..., min_length=1, description="Items to encode")
    params: EncodeParamsModel | None = Field(default=None, description="Encoding parameters")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [{"text": "Hello, world!"}, {"text": "How are you?"}],
                }
            ]
        }
    }


# Response models
class DenseVectorModel(BaseModel):
    """Dense embedding vector."""

    dims: int = Field(..., description="Vector dimensionality")
    dtype: Literal["float32", "float16", "int8", "uint8", "binary"] = Field(..., description="Data type")
    values: list[float] = Field(..., description="Vector values")


class SparseVectorModel(BaseModel):
    """Sparse embedding vector."""

    dims: int | None = Field(default=None, description="Vocabulary size")
    dtype: Literal["float32", "float16"] = Field(..., description="Data type")
    indices: list[int] = Field(..., description="Non-zero indices")
    values: list[float] = Field(..., description="Non-zero values")


class MultiVectorModel(BaseModel):
    """Multi-vector (token-level) embedding."""

    token_dims: int = Field(..., description="Dimension per token")
    num_tokens: int = Field(..., description="Number of tokens")
    dtype: Literal["float32", "float16", "int8", "uint8", "binary"] = Field(..., description="Data type")
    values: list[list[float]] = Field(..., description="Token embeddings (num_tokens x token_dims)")


class EncodeResultModel(BaseModel):
    """Single item encoding result."""

    id: str | None = Field(default=None, description="Item ID (if provided in request)")
    dense: DenseVectorModel | None = Field(default=None, description="Dense embedding")
    sparse: SparseVectorModel | None = Field(default=None, description="Sparse embedding")
    multivector: MultiVectorModel | None = Field(default=None, description="Multi-vector embedding")


class TimingInfoModel(BaseModel):
    """Request timing breakdown."""

    total_ms: float = Field(..., description="Total request time in milliseconds")
    queue_ms: float = Field(..., description="Time waiting in queue")
    tokenization_ms: float = Field(..., description="Tokenization time")
    inference_ms: float = Field(..., description="Model inference time")
    postprocessing_ms: float | None = Field(default=None, description="Postprocessing time")


class EncodeResponseModel(BaseModel):
    """Response from encode endpoint."""

    model: str = Field(..., description="Model used for encoding")
    items: list[EncodeResultModel] = Field(..., description="Encoding results for each input item")
    timing: TimingInfoModel | None = Field(default=None, description="Request timing breakdown")


# Extract endpoint models
class ExtractParamsModel(BaseModel):
    """Parameters for extract requests."""

    labels: list[str] | None = Field(default=None, description="Entity labels to extract")
    output_schema: dict[str, Any] | None = Field(default=None, description="Schema for structured extraction")
    instruction: str | None = Field(default=None, description="Task instruction")
    options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Adapter-specific options. Recognized sub-keys include "
            "'overflow_policy' (one of 'default', 'truncate_text', 'error'; "
            "default 'default') controlling how inputs exceeding the model's "
            "max_sequence_length are handled."
        ),
    )


class ExtractRequestModel(BaseModel):
    """Request body for extract endpoint."""

    items: list[ItemModel] = Field(..., min_length=1, description="Items to extract from")
    params: ExtractParamsModel | None = Field(default=None, description="Extraction parameters")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [{"text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."}],
                    "params": {"labels": ["person", "organization", "location"]},
                },
                {
                    "items": [{"text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."}],
                    "params": {
                        "labels": ["person", "organization", "location"],
                        "options": {"overflow_policy": "truncate_text"},
                    },
                },
            ]
        }
    }


class EntityModel(BaseModel):
    """Extracted entity."""

    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label/type")
    score: float = Field(..., description="Confidence score")
    start: int | None = Field(default=None, description="Start character offset")
    end: int | None = Field(default=None, description="End character offset")
    bbox: list[float] | None = Field(default=None, description="Bounding box for document entities")


class RelationModel(BaseModel):
    """Extracted relation between entities."""

    head: str = Field(..., description="Head entity text")
    tail: str = Field(..., description="Tail entity text")
    relation: str = Field(..., description="Relation type")
    score: float = Field(..., description="Confidence score")


class ClassificationModel(BaseModel):
    """Classification result."""

    label: str = Field(..., description="Classification label")
    score: float = Field(..., description="Confidence score")


class ExtractItemErrorModel(BaseModel):
    """Stable per-item extraction failure."""

    code: str = Field(..., description="Stable extraction error code")
    message: str = Field(..., description="Sanitized extraction error message")


class ExtractResultModel(BaseModel):
    """Single item extraction result."""

    id: str = Field(..., description="Item ID")
    entities: list[EntityModel] = Field(default_factory=list, description="Extracted entities")
    relations: list[RelationModel] = Field(default_factory=list, description="Extracted relations")
    classifications: list[ClassificationModel] = Field(default_factory=list, description="Classification results")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured extraction data")
    error: ExtractItemErrorModel | None = Field(default=None, description="Per-item extraction failure")


class ExtractResponseModel(BaseModel):
    """Response from extract endpoint."""

    model: str = Field(..., description="Model used for extraction")
    items: list[ExtractResultModel] = Field(..., description="Extraction results for each input item")


# Score endpoint models
class ScoreRequestModel(BaseModel):
    """Request body for score endpoint."""

    query: ItemModel = Field(..., description="Query item to score against")
    items: list[ItemModel] = Field(..., min_length=1, max_length=MAX_SCORE_ITEMS, description="Items to score")
    instruction: str | None = Field(default=None, description="Optional scoring instruction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": {"text": "What is machine learning?"},
                    "items": [
                        {"text": "Machine learning is a branch of AI..."},
                        {"text": "The weather is nice today."},
                    ],
                }
            ]
        }
    }


class ScoreEntryModel(BaseModel):
    """Single score entry."""

    item_id: str = Field(..., description="Item ID")
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Rank (0 = most relevant)")


class ScoreUsageModel(BaseModel):
    """Authoritative worker-emitted score usage."""

    input_tokens: int = Field(..., ge=0, description="Post-truncation input tokens processed")
    images: int | None = Field(default=None, ge=0, description="Images processed across query-document pairs")


class ScoreResponseModel(BaseModel):
    """Response from score endpoint."""

    model: str = Field(..., description="Model used for scoring")
    query_id: str | None = Field(default=None, description="Query ID (if provided)")
    scores: list[ScoreEntryModel] = Field(..., description="Scores sorted by relevance (descending)")
    usage: ScoreUsageModel | None = Field(default=None, description="Authoritative usage when emitted by the adapter")


# Generate endpoint models
class NativeGenerateImageModel(BaseModel):
    """One inline image on the SIE-native generate surface."""

    data: str = Field(
        ...,
        min_length=1,
        max_length=22_369_624,
        description="Canonical standard-base64 image bytes, at most 16 MiB decoded",
    )
    format: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9.+-]+$",
        description="Short media-format hint such as png or jpeg",
    )


class NativeJsonSchemaGrammarModel(BaseModel):
    """JSON Schema structured-output constraint."""

    json_schema: dict[str, Any]
    label: str | None = None
    strict: bool | None = None
    model_config = {"extra": "forbid"}


class NativeRegexGrammarModel(BaseModel):
    """Regular-expression structured-output constraint."""

    regex: str = Field(..., max_length=4 * 1024)
    label: str | None = None
    strict: bool | None = None
    model_config = {"extra": "forbid"}


class NativeEbnfGrammarModel(BaseModel):
    """EBNF structured-output constraint."""

    ebnf: str = Field(..., max_length=8 * 1024)
    label: str | None = None
    strict: bool | None = None
    model_config = {"extra": "forbid"}


class GenerateRequestModel(BaseModel):
    """Request body for the SIE-native generate endpoint."""

    prompt: str = Field(..., min_length=1, description="Prompt text to generate from")
    images: list[NativeGenerateImageModel] | None = Field(
        default=None,
        min_length=1,
        max_length=16,
        description=(
            "Optional inline images paired with prompt. The worker renders one user turn through "
            "the model's native chat template; remote URLs are not accepted."
        ),
    )
    grammar: NativeJsonSchemaGrammarModel | NativeRegexGrammarModel | NativeEbnfGrammarModel | None = Field(
        default=None,
        description="Optional structured-output grammar; exactly one of json_schema, regex, or ebnf",
    )
    max_new_tokens: int = Field(..., ge=1, description="Maximum number of tokens to generate")
    temperature: float | None = Field(default=None, ge=0, description="Sampling temperature override")
    top_p: float | None = Field(default=None, gt=0, le=1, description="Nucleus-sampling probability override")
    options: dict[str, Any] | None = Field(
        default=None,
        description="Governed generation runtime options; explicit top-level sampler fields override them",
    )
    stop: list[Annotated[str, Field(min_length=1)]] | None = Field(default=None, description="Stop sequences")
    stream: bool | None = Field(default=None, description="Return SIE-native Server-Sent Events when true")
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: float | None = Field(default=None, ge=-2, le=2, description="Presence penalty")
    seed: int | None = Field(
        default=None,
        ge=-(1 << 63),
        le=(1 << 63) - 1,
        description=(
            "Optional signed 64-bit per-request sampling seed. Reproducibility is best effort, not guaranteed, "
            "and depends on the active generation backend and deployment configuration. Non-integer or "
            "out-of-range values reject with 400 invalid_request."
        ),
        json_schema_extra={"format": "int64"},
    )
    logit_bias: dict[str, float] | None = Field(
        default=None,
        max_length=1024,
        description="Token-id string to finite sampler bias in [-100, 100]",
    )
    logprobs: bool | None = Field(
        default=None,
        description="Return per-token log probabilities; supported only with stream true",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="Alternative tokens per position; requires logprobs and stream true",
    )
    routing_key: str | None = Field(default=None, description="Optional routing hint")
    prompt_cache_key: str | None = Field(default=None, description="Optional prompt-cache routing hint")
    safety_identifier: str | None = Field(
        default=None,
        description="Sensitive client identifier; validated and dropped without logging or forwarding",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write one sentence about vector search.",
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False,
                }
            ]
        }
    }


class GenerateUsageModel(BaseModel):
    """Token usage for one generation request."""

    prompt_tokens: int = Field(..., ge=0, description="Number of prompt tokens")
    completion_tokens: int = Field(..., ge=0, description="Number of generated tokens")
    total_tokens: int = Field(..., ge=0, description="Total prompt and generated tokens")


class GenerateResponseModel(BaseModel):
    """Blocking response from the SIE-native generate endpoint."""

    model: str = Field(..., description="Model used for generation")
    text: str = Field(..., description="Generated text")
    finish_reason: str = Field(..., description="Reason generation stopped")
    usage: GenerateUsageModel = Field(..., description="Token usage")


class GenerateInputTooLongDetailModel(BaseModel):
    """Detail returned when a generation prompt exceeds the worker limit."""

    code: Literal["INPUT_TOO_LONG"] = Field(..., description="Stable error code")
    message: str = Field(..., description="Client-safe error message")
    param: str | None = Field(default=None, description="Request field that exceeded the limit")


class GenerateInputTooLongErrorResponse(BaseModel):
    """FastAPI error envelope for an oversized generation prompt."""

    detail: GenerateInputTooLongDetailModel


class GenerateModelLoadFailedDetailModel(BaseModel):
    """Detail returned for a terminal generation model-load failure."""

    code: Literal["MODEL_LOAD_FAILED"] = Field(..., description="Stable error code")
    message: str = Field(..., description="Client-safe error message")
    error_class: str = Field(..., description="Classified model-load failure category")
    permanent: bool = Field(..., description="Whether operator action is required before retrying")
    attempts: int = Field(..., ge=1, description="Number of failed model-load attempts")


class GenerateModelLoadFailedErrorResponse(BaseModel):
    """FastAPI error envelope for a terminal generation model-load failure."""

    detail: GenerateModelLoadFailedDetailModel


class GenerateChunkErrorModel(BaseModel):
    """Terminal error carried by a generation SSE event."""

    code: str = Field(..., description="Stable error code")
    message: str = Field(..., description="Client-safe error message")


class GenerateChunk(BaseModel):
    """One SIE-native Server-Sent Event from a streaming generate request."""

    request_id: str = Field(..., description="Request id shared by every event in the stream")
    seq: int = Field(..., ge=0, description="Monotonic event sequence number")
    text_delta: str = Field(..., description="Incremental generated text")
    done: bool = Field(..., description="True for the terminal event")
    finish_reason: str | None = Field(default=None, description="Termination reason on the terminal event")
    usage: GenerateUsageModel | None = Field(default=None, description="Token usage on the terminal event")
    ttft_ms: float | None = Field(default=None, ge=0, description="Time to first token on the terminal event")
    logprobs: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-token log probabilities aligned with text_delta",
    )
    error: GenerateChunkErrorModel | None = Field(default=None, description="Terminal generation error")
