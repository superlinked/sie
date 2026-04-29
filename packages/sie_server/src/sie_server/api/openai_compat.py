"""OpenAI-compatible embeddings endpoint for SIE Server.

POST /v1/embeddings - Generate embeddings using OpenAI's API format.

This enables zero-friction migration from OpenAI, Azure OpenAI, or any
OpenAI-compatible embedding service. Works with LangChain's OpenAIEmbeddings
class out of the box:

    embeddings = OpenAIEmbeddings(base_url="http://localhost:8080/v1")

Per roadmap Project 11.0: OpenAI-Compatible API.
"""

from __future__ import annotations

import base64
import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from sie_server.api.helpers import extract_request_context
from sie_server.api.validation import validate_machine_profile_header
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.worker import QueueFullError
from sie_server.observability.metrics import record_request
from sie_server.observability.tracing import tracer
from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# OpenAI-compatible request/response types


# Type alias for OpenAI input formats
# OpenAI accepts: string, array of strings, array of tokens, or array of token arrays
OpenAIInput = str | list[str] | list[int] | list[list[int]]


class OpenAIEmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request.

    See: https://platform.openai.com/docs/api-reference/embeddings
    """

    model_config = ConfigDict(extra="ignore")  # Ignore unknown fields

    model: Annotated[str, Field(description="Model ID to use for embedding")]
    input: Annotated[
        OpenAIInput,
        Field(description="Input text(s) or token array(s) to embed."),
    ]
    encoding_format: Annotated[
        Literal["float", "base64"] | None,
        Field(default="float", description="Format for embeddings: 'float' or 'base64'"),
    ]
    dimensions: Annotated[
        int | None,
        Field(default=None, description="Number of dimensions (not supported by SIE, ignored)"),
    ]
    user: Annotated[
        str | None,
        Field(default=None, description="User ID for tracking (ignored by SIE)"),
    ]


class OpenAIEmbeddingData(BaseModel):
    """Single embedding result in OpenAI format."""

    model_config = ConfigDict(extra="forbid")

    object: Annotated[Literal["embedding"], Field(default="embedding")]
    embedding: Annotated[list[float] | str, Field(description="Embedding vector (floats or base64)")]
    index: Annotated[int, Field(description="Index in the input array")]


class OpenAIUsage(BaseModel):
    """Token usage information."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: Annotated[int, Field(description="Number of tokens in the input")]
    total_tokens: Annotated[int, Field(description="Total tokens (same as prompt_tokens for embeddings)")]


class OpenAIEmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    model_config = ConfigDict(extra="forbid")

    object: Annotated[Literal["list"], Field(default="list")]
    data: Annotated[list[OpenAIEmbeddingData], Field(description="Embedding results")]
    model: Annotated[str, Field(description="Model used")]
    usage: Annotated[OpenAIUsage, Field(description="Token usage")]


def _encode_base64(embedding: np.ndarray) -> str:
    """Encode embedding as base64 string (little-endian float32).

    OpenAI's base64 format uses little-endian float32.
    """
    # Ensure float32
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return base64.b64encode(embedding.tobytes()).decode("ascii")


def _estimate_tokens(texts: list[str]) -> int:
    """Estimate token count from text length.

    This is a rough estimate. For accurate counts, we'd need the actual tokenizer.
    Using ~4 chars per token as a reasonable approximation.
    """
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // 4)


def _normalize_input(input_data: OpenAIInput, registry: object, model: str) -> tuple[list[str], int]:
    """Normalize OpenAI input format to list of strings.

    OpenAI accepts:
    - str: single text
    - list[str]: multiple texts
    - list[int]: single token array
    - list[list[int]]: multiple token arrays

    Args:
        input_data: Raw input from request
        registry: Model registry (for tokenizer access)
        model: Model name

    Returns:
        Tuple of (list of texts, token count)
    """
    # Single string
    if isinstance(input_data, str):
        return [input_data], _estimate_tokens([input_data])

    # Empty list
    if not input_data:
        return [], 0

    # Check if it's a token array (list[int]) or list of token arrays (list[list[int]])
    first = input_data[0]

    if isinstance(first, str):
        # list[str] - multiple texts
        return list(input_data), _estimate_tokens(input_data)  # type: ignore

    if isinstance(first, int):
        # list[int] - single token array, decode it
        token_count = len(input_data)
        text = _decode_tokens(input_data, registry, model)  # type: ignore
        return [text], token_count

    if isinstance(first, list):
        # list[list[int]] - multiple token arrays
        texts = []
        token_count = 0
        for tokens in input_data:
            if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                texts.append(_decode_tokens(tokens, registry, model))
                token_count += len(tokens)
            else:
                # Unexpected format
                texts.append(str(tokens))
        return texts, token_count

    # Fallback: convert to string
    return [str(input_data)], 1


def _decode_tokens(tokens: list[int], registry: object, model: str) -> str:
    """Decode token IDs back to text using the model's tokenizer.

    Args:
        tokens: List of token IDs
        registry: Model registry
        model: Model name

    Returns:
        Decoded text string
    """
    try:
        preprocessor_registry = registry.preprocessor_registry  # type: ignore
        if preprocessor_registry.has_preprocessor(model, "text"):
            tokenizer = preprocessor_registry.get_tokenizer(model)
            if tokenizer is not None:
                return tokenizer.decode(tokens, skip_special_tokens=True)
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Token decoding failed for model %s: %s", model, e)

    # Fallback: can't decode, return placeholder
    # This happens if the model doesn't have a registered tokenizer
    logger.warning("Cannot decode tokens for model %s, using placeholder", model)
    return f"[{len(tokens)} tokens]"


async def _load_model_if_needed(registry: object, model: str, device: str, span: object) -> None:
    """Load model if not already loaded."""
    if not registry.is_loaded(model):  # type: ignore
        try:
            logger.info("Loading model %s on device %s", model, device)
            await registry.load_async(model, device=device)  # type: ignore
        except Exception as e:
            logger.exception("Failed to load model %s", model)
            span.set_attribute("error", "model_load_failed")  # type: ignore
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "model_not_available",
                        "message": f"Failed to load model: {e}",
                        "type": "server_error",
                    }
                },
            ) from e


def _build_embeddings_response(
    results: list[dict],
    texts: list[str],
    model: str,
    encoding_format: str,
    token_count: int | None = None,
) -> OpenAIEmbeddingResponse:
    """Build OpenAI-format response from encoding results.

    Args:
        results: Encoding results from adapter
        texts: Input texts
        model: Model name
        encoding_format: "float" or "base64"
        token_count: Known token count (from token input), or None to estimate
    """
    embeddings_data: list[OpenAIEmbeddingData] = []

    for i, result in enumerate(results):
        dense = result.get("dense")
        if dense is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "no_embedding",
                        "message": "Model did not return dense embedding",
                        "type": "server_error",
                    }
                },
            )

        # Get the raw numpy array from dense result
        embedding_values = dense if isinstance(dense, np.ndarray) else dense.get("values", dense)
        if isinstance(embedding_values, np.ndarray):
            if encoding_format == "base64":
                embedding: list[float] | str = _encode_base64(embedding_values)
            else:
                embedding = embedding_values.tolist()
        else:
            # Already a list
            embedding = embedding_values

        embeddings_data.append(
            OpenAIEmbeddingData(
                object="embedding",
                embedding=embedding,
                index=i,
            )
        )

    # Use provided token count or estimate
    if token_count is None or token_count == 0:
        token_count = _estimate_tokens(texts)

    return OpenAIEmbeddingResponse(
        object="list",
        data=embeddings_data,
        model=model,
        usage=OpenAIUsage(
            prompt_tokens=token_count,
            total_tokens=token_count,
        ),
    )


@router.post(
    "/embeddings",
    responses={
        200: {"description": "Embeddings generated successfully"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        502: {
            "description": (
                "Terminal model-load failure (MODEL_LOAD_FAILED). "
                "Carried in the ``detail`` envelope: ``{code, message, "
                "error_class, permanent, attempts}``. No ``Retry-After`` "
                "header — clients MUST NOT auto-retry. See sie-test#85."
            ),
        },
        503: {"description": "Service unavailable"},
    },
)
async def create_embeddings(
    request: OpenAIEmbeddingRequest,
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> OpenAIEmbeddingResponse:
    """Create embeddings using OpenAI-compatible API.

    This endpoint is compatible with OpenAI's /v1/embeddings API, allowing
    drop-in replacement for any OpenAI SDK or client.

    Args:
        request: OpenAI-format embedding request.
        http_request: FastAPI request (for app state).
        x_machine_profile: Machine profile header for routing validation.

    Returns:
        OpenAI-format embedding response with embeddings and usage info.
    """
    # Validate machine profile header
    validate_machine_profile_header(x_machine_profile)

    model = request.model
    encoding_format = request.encoding_format or "float"

    with tracer.start_as_current_span("openai_embeddings") as span:
        span.set_attribute("model", model)

        registry = http_request.app.state.registry
        ctx = extract_request_context(http_request, model, registry)

        # Check if model exists first (needed for token decoding)
        if not registry.has_model(model):
            span.set_attribute("error", "model_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "model_not_found",
                        "message": f"Model '{model}' not found",
                        "type": "invalid_request_error",
                    }
                },
            )

        # Check if model is being unloaded
        if registry.is_unloading(model):
            span.set_attribute("error", "model_unloading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "model_not_available",
                        "message": f"Model '{model}' is unloading",
                        "type": "server_error",
                    }
                },
            )

        # Normalize input (handles strings, token arrays, etc.)
        texts, token_count = _normalize_input(request.input, registry, model)

        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_request_error", "message": "Input cannot be empty"},
            )

        span.set_attribute("batch_size", len(texts))

        # Load model if needed
        device = registry.device
        await _load_model_if_needed(registry, model, device, span)

        # Get config and convert texts to SIE Items
        config = registry.get_config(model)
        items = [Item(text=text) for text in texts]

        # Run encoding
        try:
            results, timing = await EncodePipeline.run_encode(
                registry=registry,
                model=model,
                items=items,
                output_types=["dense"],
                instruction=None,
                config=config,
                is_query=False,
                options={},
            )
        except QueueFullError as e:
            span.set_attribute("error", "queue_full")
            record_request(
                model=model,
                endpoint="embeddings",
                status="queue_full",
                request_id=ctx.request_id,
                api_key=ctx.api_key,
                queue_depth=ctx.queue_depth,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "server_overloaded",
                        "message": str(e),
                        "type": "server_error",
                    }
                },
            ) from e
        except Exception as e:
            logger.exception("Inference error for model %s", model)
            span.set_attribute("error", "inference_error")
            record_request(
                model=model,
                endpoint="embeddings",
                status="error",
                request_id=ctx.request_id,
                api_key=ctx.api_key,
                queue_depth=ctx.queue_depth,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "inference_error",
                        "message": f"Inference error: {e}",
                        "type": "server_error",
                    }
                },
            ) from e

        # Build and return response
        record_request(
            model=model,
            endpoint="embeddings",
            status="success",
            timing=timing,
            request_id=ctx.request_id,
            api_key=ctx.api_key,
            queue_depth=ctx.queue_depth,
        )
        return _build_embeddings_response(results, texts, model, encoding_format, token_count)
