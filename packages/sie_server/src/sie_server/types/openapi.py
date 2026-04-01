"""Pydantic models for OpenAPI documentation.

These models mirror the TypedDict types in sie_server.types but are Pydantic BaseModel
classes that FastAPI uses to generate OpenAPI schemas in Swagger UI.

The actual request/response handling uses TypedDict for zero overhead, while these
models provide rich documentation with descriptions and examples.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Request models
class ImageInputModel(BaseModel):
    """Image input for multimodal models."""

    data: bytes = Field(..., description="Image data as bytes")
    format: str | None = Field(default=None, description="Image format hint: 'jpeg', 'png', etc.")


class ItemModel(BaseModel):
    """A single item to encode."""

    id: str | None = Field(default=None, description="Optional identifier for this item. Returned in response.")
    text: str | None = Field(default=None, description="Text content to encode", examples=["Hello, world!"])
    images: list[ImageInputModel] | None = Field(default=None, description="Images for multimodal models")
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
    options: dict[str, Any] | None = Field(default=None, description="Adapter-specific options")


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
                }
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


class ExtractResultModel(BaseModel):
    """Single item extraction result."""

    id: str = Field(..., description="Item ID")
    entities: list[EntityModel] = Field(default_factory=list, description="Extracted entities")
    relations: list[RelationModel] = Field(default_factory=list, description="Extracted relations")
    classifications: list[ClassificationModel] = Field(default_factory=list, description="Classification results")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured extraction data")


class ExtractResponseModel(BaseModel):
    """Response from extract endpoint."""

    model: str = Field(..., description="Model used for extraction")
    items: list[ExtractResultModel] = Field(..., description="Extraction results for each input item")


# Score endpoint models
class ScoreRequestModel(BaseModel):
    """Request body for score endpoint."""

    query: ItemModel = Field(..., description="Query item to score against")
    items: list[ItemModel] = Field(..., min_length=1, description="Items to score")
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


class ScoreResponseModel(BaseModel):
    """Response from score endpoint."""

    model: str = Field(..., description="Model used for scoring")
    query_id: str | None = Field(default=None, description="Query ID (if provided)")
    scores: list[ScoreEntryModel] = Field(..., description="Scores sorted by relevance (descending)")
