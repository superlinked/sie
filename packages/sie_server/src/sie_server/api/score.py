import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from sie_server.api.helpers import (
    InferenceErrorHandler,
    ModelStateChecker,
    RequestParser,
    ResponseBuilder,
    oom_retry_after_from_registry,
)
from sie_server.api.options import resolve_runtime_options
from sie_server.api.serialization import MsgPackResponse
from sie_server.api.validation import validate_machine_profile_header
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.score_cost import build_score_prepared_items_timed
from sie_server.core.worker import QueueFullError, WorkerResult
from sie_server.observability.tracing import tracer
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled
from sie_server.types.inputs import Item
from sie_server.types.openapi import ScoreResponseModel
from sie_server.types.requests import ScoreRequest
from sie_server.types.responses import ErrorCode, ScoreEntry, ScoreResponse, ScoreUsage

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["score"])


def _build_response(
    model: str,
    query_id: str | None,
    items: list[Any],
    scores: list[float],
    usage: ScoreUsage | None = None,
) -> ScoreResponse:
    """Build ScoreResponse from adapter scores.

    Scores are sorted by relevance (descending) and ranked.
    """
    # Create (index, item_id, score) tuples
    scored_items = []
    for i, score in enumerate(scores):
        item_id = items[i].id if items[i].id is not None else f"item-{i}"
        scored_items.append((i, item_id, score))

    # Sort by score descending
    scored_items.sort(key=lambda x: x[2], reverse=True)

    # Build ScoreEntry list with ranks
    entries = []
    for rank, (_, item_id, score) in enumerate(scored_items):
        entries.append(
            ScoreEntry(
                item_id=item_id,
                score=score,
                rank=rank,
            )
        )

    response = ScoreResponse(
        model=model,
        query_id=query_id,
        scores=entries,
    )
    if usage is not None:
        response["usage"] = usage
    return response


def score_usage_from_output(output: ScoreOutput) -> ScoreUsage | None:
    """Return exact post-tokenization score usage, never a character estimate."""
    if output.input_token_counts is None:
        return None
    usage = ScoreUsage(input_tokens=sum(output.input_token_counts))
    if output.input_image_counts is not None:
        usage["images"] = sum(output.input_image_counts)
    return usage


async def _score_via_worker(
    registry: "ModelRegistry",
    model: str,
    query: Item,
    items: list[Item],
    *,
    instruction: str | None = None,
    options: dict[str, Any] | None = None,
) -> WorkerResult:
    """Score using the async worker with dynamic batching.

    This path provides better throughput under concurrent load by batching
    (query, doc) pairs from different requests together.

    Args:
        registry: ModelRegistry instance.
        model: Model name.
        query: Query item.
        items: Items to score against the query.
        instruction: Optional instruction.
        options: Runtime options (resolved from profile + overrides).

    Returns:
        WorkerResult containing score results and timing information.
    """
    # Build score PreparedItems + tokenization timing via the shared helper
    # (same source of truth as the sidecar IPC path in queue_executor).
    prepared_items, timing = build_score_prepared_items_timed(query, items)

    # Start worker if not running
    worker = await registry.start_worker(model)

    # Submit to worker and await result
    future = await worker.submit_score(
        prepared_items=prepared_items,
        query=query,
        items=items,
        instruction=instruction,
        options=options,
        timing=timing,
    )

    return await future


@router.post(
    "/score/{model:path}",
    response_model=None,  # We handle serialization manually for content negotiation
    responses={
        200: {
            "description": "Scores computed successfully",
            "model": ScoreResponseModel,
            "content": {
                "application/msgpack": {},
            },
        },
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
        503: {"description": "Model not loaded or service unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ScoreRequestModel"},
                },
                "application/msgpack": {
                    "schema": {"$ref": "#/components/schemas/ScoreRequestModel"},
                },
            },
        },
    },
)
async def score(
    model: str,
    http_request: Request,
    accept: Annotated[str | None, Header()] = None,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> MsgPackResponse | JSONResponse:
    """Score items against a query using a reranker model.

    Supports both msgpack and JSON request bodies (Content-Type header).
    Returns msgpack by default, JSON if Accept header requests it.

    Args:
        model: Model name to use for scoring.
        http_request: FastAPI request object (for body and app state).
        accept: Accept header for response content negotiation.
        x_machine_profile: Machine profile header for routing validation.

    Returns:
        ScoreResponse with scores sorted by relevance (descending).
        Format depends on Accept header: msgpack (default) or JSON.

    Raises:
        HTTPException: 400 for invalid input or profile mismatch, 404 if model not found,
            503 if not loaded.
    """
    # Validate machine profile header against worker identity (catches routing errors early)
    validate_machine_profile_header(x_machine_profile)

    # Start tracing span for score operation
    with tracer.start_as_current_span("score") as span:
        span.set_attribute("model", model)
        if x_machine_profile:
            span.set_attribute("machine_profile", x_machine_profile)

        request = await RequestParser.parse(http_request, ScoreRequest)

        # Set span attributes from request
        span.set_attribute("batch_size", len(request.items))

        registry = http_request.app.state.registry
        device = registry.device

        # Validate model state using helper (split to check capability before loading)
        model_checker = ModelStateChecker(registry, model, span)
        model_checker.check_exists()

        # Check model config supports scoring (before loading gate — fail fast)
        config = registry.get_config(model)
        if config.tasks.score is None:
            span.set_attribute("error", "unsupported_operation")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": f"Model '{model}' does not support scoring. "
                    f"Use an encoder model with /v1/encode instead, or use a reranker model.",
                },
            )

        # Continue model state validation
        model_checker.check_not_unloading()
        model_checker.check_not_loading()
        await model_checker.ensure_loaded(device)

        # Resolve profile and merge runtime options (outside inference try/except
        # so ValueError from invalid profiles returns 400, not 500)
        instruction = request.instruction
        profile_name = str(request.options.get("profile") or "default") if request.options else "default"
        options = resolve_runtime_options(config, request.options, span)

        # Request-level instruction takes precedence; fall back to profile instruction
        if instruction is None:
            instruction = options.get("instruction")

        query = request.query
        items = request.items

        # Score using worker with batching
        error_handler = InferenceErrorHandler(
            model,
            "score",
            span,
            oom_retry_after_s=oom_retry_after_from_registry(registry),
            profile=profile_name,
            item_count=len(items),
        )
        try:
            worker_result = await _score_via_worker(
                registry,
                model,
                query,
                items,
                instruction=instruction,
                options=options,
            )
            # Format typed output to extract scores
            score_output: ScoreOutput = worker_result.output  # type: ignore
            scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]
            timing = worker_result.timing
        except QueueFullError as e:
            raise error_handler.handle_queue_full(e) from e
        except ValueError as e:
            raise error_handler.handle_value_error(e) from e
        except Exception as e:
            raise error_handler.handle_inference_error(e) from e

        # Build response
        query_id = query.id
        response = _build_response(model, query_id, items, scores, score_usage_from_output(score_output))

        if worker_telemetry_enabled():
            units = None
            token_counts = score_output.input_token_counts
            if (
                isinstance(token_counts, list)
                and len(token_counts) == len(items)
                and all(isinstance(count, int) and not isinstance(count, bool) for count in token_counts)
            ):
                units = {"input_tokens": sum(token_counts)}
            worker_telemetry().item_completed(
                operation="score",
                outcome="success",
                model=model,
                profile=profile_name,
                duration_s=timing.total_ms / 1000.0,
                item_count=len(items),
                tokenization_s=timing.tokenization_ms / 1000.0,
                inference_s=timing.inference_ms / 1000.0,
                postprocessing_s=timing.postprocessing_ms / 1000.0,
                units=units,
            )

        # Build response headers and return
        headers = ResponseBuilder.build_headers(timing)
        return ResponseBuilder.build_response(response, accept, headers)
