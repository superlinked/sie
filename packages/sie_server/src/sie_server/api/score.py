import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from sie_server.api.helpers import (
    InferenceErrorHandler,
    ModelStateChecker,
    RequestParser,
    ResponseBuilder,
    extract_request_context,
    oom_retry_after_from_registry,
)
from sie_server.api.options import resolve_runtime_options
from sie_server.api.serialization import MsgPackResponse
from sie_server.api.validation import validate_machine_profile_header
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.prepared import ScorePreparedItem
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import QueueFullError, WorkerResult
from sie_server.observability.metrics import record_request
from sie_server.observability.tracing import tracer
from sie_server.types.inputs import Item
from sie_server.types.openapi import ScoreResponseModel
from sie_server.types.requests import ScoreRequest
from sie_server.types.responses import ErrorCode, ScoreEntry, ScoreResponse

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["score"])


def _build_response(
    model: str,
    query_id: str | None,
    items: list[Any],
    scores: list[float],
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

    return ScoreResponse(
        model=model,
        query_id=query_id,
        scores=entries,
    )


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
    # Create timing tracker for this request
    timing = RequestTiming()

    # Get query text length for cost calculation
    query_text = query.text
    query_len = len(query_text) if query_text else 0

    # Create PreparedItems for batching (cost = query + doc char count)
    timing.start_tokenization()  # Using tokenization timing for prep phase
    prepared_items = []
    for i, item in enumerate(items):
        item_text = item.text
        doc_len = len(item_text) if item_text else 0
        prepared = ScorePreparedItem(
            cost=query_len + doc_len,
            original_index=i,
        )
        prepared_items.append(prepared)
    timing.end_tokenization()

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

        # Extract request context for structured logging
        ctx = extract_request_context(http_request, model, registry)

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
            ctx=ctx,
            oom_retry_after_s=oom_retry_after_from_registry(registry),
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
        response = _build_response(model, query_id, items, scores)

        # Record successful request
        record_request(
            model=model,
            endpoint="score",
            status="success",
            timing=timing,
            request_id=ctx.request_id,
            api_key=ctx.api_key,
            queue_depth=ctx.queue_depth,
        )

        # Build response headers and return
        headers = ResponseBuilder.build_headers(timing)
        return ResponseBuilder.build_response(response, accept, headers)
