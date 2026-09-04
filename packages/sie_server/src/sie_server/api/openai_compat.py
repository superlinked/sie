"""OpenAI-compatible embeddings endpoint for SIE Server.

POST /v1/embeddings - Generate embeddings using OpenAI's API format.

This enables zero-friction migration from OpenAI, Azure OpenAI, or any
OpenAI-compatible embedding service. Works with LangChain's OpenAIEmbeddings
class out of the box:

    embeddings = OpenAIEmbeddings(base_url="http://localhost:8080/v1")

This module implements the OpenAI-compatible embeddings API surface.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from sie_server.api.helpers import (
    WORKER_DRAINED_RETRY_AFTER_S,
    ModelStateChecker,
    check_sdk_version,
    oom_retry_after_from_registry,
    openai_error_response,
)
from sie_server.api.validation import validate_machine_profile_header
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.model_suggestions import suggestion_suffix
from sie_server.core.oom import is_oom_error
from sie_server.core.worker import QueueFullError
from sie_server.core.worker.types import WorkerDrainedError
from sie_server.observability.tracing import tracer
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled
from sie_server.types.inputs import Item
from sie_server.types.openapi import (
    OpenAIEmbeddingsErrorResponse,
    OpenAIEmbeddingsModelLoadFailedErrorResponse,
)
from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

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
        Field(
            default=None,
            description=(
                "Requested embedding width. SIE always returns the model's native dense "
                "width, so this is accepted only when it equals that width; any other "
                "value is rejected with 400 `unsupported_field` rather than ignored."
            ),
        ),
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


def _openai_state_error(error: HTTPException) -> HTTPException:
    """Re-wrap a native model-state ``detail`` in this surface's OpenAI envelope.

    :class:`ModelStateChecker` raises the SIE-native ``{code, message, ...}``
    detail shape served by the native routes. This surface keeps errors
    OpenAI-parseable, so rebuild the detail as ``{"error": {...}}`` while
    preserving the SIE-native ``code`` (``MODEL_LOADING`` /
    ``MODEL_LOAD_FAILED``) plus auxiliary fields (``error_class``,
    ``permanent``, ``attempts``) — the SDK's ``get_error_code`` and
    ``raise_if_model_load_failed`` branch on exactly those. Headers are
    preserved verbatim: ``Retry-After`` presence/absence carries the
    retryability contract (present on 503 ``MODEL_LOADING``, absent on the
    terminal 502 ``MODEL_LOAD_FAILED``).
    """
    detail = error.detail if isinstance(error.detail, dict) else {}
    if "error" in detail:
        return error
    inner: dict[str, object] = dict(detail)
    inner.setdefault("message", "service unavailable")
    inner["type"] = "server_error"
    return HTTPException(status_code=error.status_code, detail={"error": inner}, headers=error.headers)


def _validate_dimensions(requested: int | None, registry: ModelRegistry, model: str) -> None:
    """Reject a ``dimensions`` value this model cannot honour.

    In OpenAI's API ``dimensions`` truncates the embedding (Matryoshka), so a
    client migrating from ``text-embedding-3-*`` may legitimately send it. SIE
    always returns the model's native dense width. Ignoring the field — as this
    endpoint used to — means such a client silently receives vectors of a width
    it never asked for and writes them into a vector store. That surfaces much
    later, either as a dimension error on insert or, worse, as a silently
    wrong index built against the wrong width.

    Failing loudly matches ``/v1/completions``, which already rejects fields it
    cannot honour with ``400 unsupported_field``. An exact match is accepted so
    a client that simply pins its model's real width keeps working.

    Raises:
        HTTPException: 400 when the value cannot be honoured.
    """
    if requested is None:
        return

    native = registry.get_config(model).dims.get("dense")
    if native is not None and requested == native:
        return

    detail = (
        f"'dimensions' is not supported by this endpoint: model '{model}' returns "
        f"{native}-dimensional embeddings and SIE does not truncate them."
        if native is not None
        else f"'dimensions' is not supported by this endpoint: model '{model}' declares no dense embedding width."
    )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": {
                "code": "unsupported_field",
                "message": f"{detail} Omit 'dimensions', or set it to {native}." if native is not None else detail,
                "type": "invalid_request_error",
                "param": "dimensions",
            }
        },
    )


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
    response_model=OpenAIEmbeddingResponse,
    responses={
        200: {"description": "Embeddings generated successfully"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        502: {
            "description": (
                "Terminal model-load failure (MODEL_LOAD_FAILED). "
                "Carried in the top-level OpenAI ``error`` envelope, whose "
                "object adds the model-load extras ``error_class``, "
                "``permanent``, and ``attempts`` to ``{message, type, param, "
                "code}``. No ``Retry-After`` header — clients MUST NOT "
                "auto-retry."
            ),
            "model": OpenAIEmbeddingsModelLoadFailedErrorResponse,
        },
        503: {
            "description": (
                "Service unavailable (retryable). A cold model starts a "
                "background load and returns ``MODEL_LOADING`` with a "
                "``Retry-After`` header immediately instead of blocking the "
                "request — clients should retry after the indicated delay. "
                "Also returned while unloading, on a full queue, and under "
                "transient resource exhaustion. The body is the top-level "
                "OpenAI ``error`` envelope."
            ),
            "model": OpenAIEmbeddingsErrorResponse,
            "headers": {
                "Retry-After": {
                    "description": "Seconds to wait before retrying the MODEL_LOADING request.",
                    "schema": {"type": "integer", "minimum": 1},
                }
            },
        },
    },
)
async def create_embeddings(
    request: OpenAIEmbeddingRequest,
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> OpenAIEmbeddingResponse | JSONResponse:
    """Create embeddings using OpenAI-compatible API.

    This endpoint is compatible with OpenAI's /v1/embeddings API, allowing
    drop-in replacement for any OpenAI SDK or client. Route-generated errors
    are emitted as top-level OpenAI ``{"error": {...}}`` envelopes, matching
    ``/v1/completions``. The one exception is 422 request-body validation,
    which still returns FastAPI's ``HTTPValidationError`` ``{"detail": [...]}``
    shape (out of scope for this route's error handling; known residual).

    Args:
        request: OpenAI-format embedding request.
        http_request: FastAPI request (for app state).
        x_machine_profile: Machine profile header for routing validation.

    Returns:
        OpenAI-format embedding response with embeddings and usage info.
    """
    try:
        return await _create_embeddings(request, http_request, x_machine_profile)
    except HTTPException as exc:
        return openai_error_response(exc)


async def _create_embeddings(
    request: OpenAIEmbeddingRequest,
    http_request: Request,
    x_machine_profile: str | None,
) -> OpenAIEmbeddingResponse:
    # Validate machine profile header
    validate_machine_profile_header(x_machine_profile)
    check_sdk_version(http_request)

    model = request.model
    encoding_format = request.encoding_format or "float"

    with tracer.start_as_current_span("openai_embeddings") as span:
        span.set_attribute("model", model)

        registry = http_request.app.state.registry

        # Check if model exists first (needed for token decoding)
        if not registry.has_model(model):
            span.set_attribute("error", "model_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "model_not_found",
                        "message": f"Model '{model}' not found{suggestion_suffix(model, registry.model_names)}",
                        "type": "invalid_request_error",
                    }
                },
            )

        # Validated before any load work: an unhonourable 'dimensions' must not
        # cost the caller a cold model load first.
        _validate_dimensions(request.dimensions, registry, model)

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
                detail={
                    "error": {
                        "code": "invalid_request",
                        "message": "Input cannot be empty",
                        "type": "invalid_request_error",
                        "param": "input",
                    }
                },
            )

        span.set_attribute("batch_size", len(texts))

        # Model load states: mirror the native routes' ModelStateChecker
        # contract instead of blocking the request on a cold load. A recorded
        # terminal failure short-circuits as 502 MODEL_LOAD_FAILED with no
        # Retry-After (and, critically, no re-triggered doomed load); a cold
        # model kicks off a background load and returns 503 MODEL_LOADING +
        # Retry-After immediately so clients retry instead of hanging. This is
        # single-node parity with the gateway's /v1/embeddings behavior.
        device = registry.device
        checker = ModelStateChecker(registry, model, span)
        try:
            checker.check_not_failed()
            checker.check_not_loading()
            await checker.ensure_loaded(device)
        except HTTPException as error:
            raise _openai_state_error(error) from error

        # Get config and convert texts to SIE Items
        config = registry.get_config(model)
        items = [Item(text=text) for text in texts]

        # Run encoding
        telemetry_enabled = worker_telemetry_enabled()
        inference_started = time.perf_counter() if telemetry_enabled else None
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
            if inference_started is not None:
                worker_telemetry().item_completed(
                    operation="embeddings",
                    outcome="retry",
                    model=model,
                    profile="default",
                    duration_s=time.perf_counter() - inference_started,
                    item_count=len(items),
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
        except WorkerDrainedError as e:
            # The model was evicted while this request sat in the worker
            # queue, so it never ran. Mirror the native endpoints' retryable
            # 503 MODEL_LOADING — in the OpenAI envelope — so the SDK retries
            # (and re-triggers the load) instead of seeing a terminal 500.
            logger.info("Embeddings request drained on eviction for model %s: %s", model, e)
            span.set_attribute("error", "model_loading")
            if inference_started is not None:
                worker_telemetry().item_completed(
                    operation="embeddings",
                    outcome="retry",
                    model=model,
                    profile="default",
                    duration_s=time.perf_counter() - inference_started,
                    item_count=len(items),
                )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": ErrorCode.MODEL_LOADING.value,
                        "message": f"Model '{model}' was evicted before this request ran, please retry",
                        "type": "server_error",
                    }
                },
                headers={"Retry-After": str(WORKER_DRAINED_RETRY_AFTER_S)},
            ) from e
        except Exception as e:
            if is_oom_error(e):
                # Transient memory pressure (worker ResourceExhaustedError after
                # recovery is exhausted, or a raw CUDA/MPS OOM). Mirror the
                # native endpoints' 503 RESOURCE_EXHAUSTED + Retry-After — in the
                # OpenAI envelope — so the SDK auto-retries instead of treating
                # it as a terminal 500. See #1604.
                logger.warning("Embeddings OOM for model %s, returning 503 RESOURCE_EXHAUSTED: %s", model, e)
                span.set_attribute("error", "resource_exhausted")
                if inference_started is not None:
                    worker_telemetry().item_completed(
                        operation="embeddings",
                        outcome="retry",
                        model=model,
                        profile="default",
                        duration_s=time.perf_counter() - inference_started,
                        item_count=len(items),
                    )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": {
                            "code": ErrorCode.RESOURCE_EXHAUSTED.value,
                            "message": f"Embeddings temporarily unavailable due to resource pressure: {e}",
                            "type": "server_error",
                        }
                    },
                    headers={"Retry-After": str(oom_retry_after_from_registry(registry))},
                ) from e
            logger.exception("Inference error for model %s", model)
            span.set_attribute("error", "inference_error")
            if inference_started is not None:
                worker_telemetry().item_completed(
                    operation="embeddings",
                    outcome="error",
                    model=model,
                    profile="default",
                    duration_s=time.perf_counter() - inference_started,
                    item_count=len(items),
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "inference_error",
                        "message": "internal error during embeddings",
                        "type": "server_error",
                    }
                },
            ) from e

        if telemetry_enabled:
            units = None
            if timing.input_token_counts is not None:
                counts = timing.input_token_counts
                if len(counts) == len(items) and all(
                    isinstance(count, int) and not isinstance(count, bool) for count in counts
                ):
                    units = {"input_tokens": sum(counts)}
            worker_telemetry().item_completed(
                operation="embeddings",
                outcome="success",
                model=model,
                profile="default",
                duration_s=timing.total_ms / 1000.0,
                item_count=len(items),
                tokenization_s=timing.tokenization_ms / 1000.0,
                inference_s=timing.inference_ms / 1000.0,
                postprocessing_s=timing.postprocessing_ms / 1000.0,
                units=units,
            )
        return _build_embeddings_response(results, texts, model, encoding_format, token_count)
