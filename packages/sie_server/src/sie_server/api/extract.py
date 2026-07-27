import logging
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from sie_server.adapters.errors import InputTooLongError
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
from sie_server.core.extract_cost import build_extract_prepared_items
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import QueueFullError, WorkerResult
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.observability.tracing import tracer
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled
from sie_server.types.inputs import Item
from sie_server.types.openapi import ExtractResponseModel
from sie_server.types.requests import ExtractRequest
from sie_server.types.responses import (
    Classification,
    DetectedObject,
    Entity,
    ErrorCode,
    ExtractItemErrorDetail,
    ExtractResponse,
    ExtractResult,
    Relation,
)

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["extract"])


async def _extract_via_worker(
    registry: "ModelRegistry",
    model: str,
    items: list[Item],
    *,
    labels: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
    instruction: str | None = None,
    options: dict[str, Any] | None = None,
) -> WorkerResult:
    """Extract using the async worker with dynamic batching.

    This path provides better throughput under concurrent load by batching
    requests together. Items with the same (labels, instruction, options) config
    can be batched together for efficient GPU utilization.

    For vision models (Florence-2, Donut), preprocessing runs on CPU thread pool
    to overlap with GPU inference.

    Args:
        registry: ModelRegistry instance.
        model: Model name.
        items: Items to extract from.
        labels: Entity types to extract.
        output_schema: Optional schema for structured extraction.
        instruction: Optional instruction.
        options: Adapter options to override model config defaults.

    Returns:
        WorkerResult containing extraction results and timing information.
    """
    # Create timing tracker for this request
    timing = RequestTiming()

    # Check whether this model owns media preprocessing.
    # Use try/except for safety in tests where registry might be mocked
    preprocessor_registry = getattr(registry, "preprocessor_registry", None)
    config = None
    try:
        if hasattr(registry, "get_config"):
            config = registry.get_config(model)
    except (AttributeError, KeyError):
        pass

    timing.start_tokenization()  # Using tokenization timing for prep phase

    # Use exact booleans so MagicMock registries in narrow unit tests do not opt in.
    # Note: Check for real PreprocessorRegistry (not MagicMock which has all attributes)
    use_preprocessor = False
    has_audio_preprocessor = False
    has_image_preprocessor = False
    if preprocessor_registry is not None and config is not None:
        try:
            has_audio_preprocessor = preprocessor_registry.has_preprocessor(model, "audio") is True
            has_image_preprocessor = preprocessor_registry.has_preprocessor(model, "image") is True
            use_preprocessor = has_audio_preprocessor or has_image_preprocessor
        except (AttributeError, TypeError):
            pass

    if use_preprocessor and preprocessor_registry is not None:
        has_audio_input = any(item.audio is not None for item in items)
        has_images = any(item.images for item in items)
        try:
            has_text_preprocessor = preprocessor_registry.has_preprocessor(model, "text") is True
        except (AttributeError, TypeError):
            has_text_preprocessor = False
        supported_input = (
            (has_audio_input and has_audio_preprocessor)
            or (has_images and has_image_preprocessor)
            or (any(item.text is not None for item in items) and has_text_preprocessor)
        )
        if not supported_input:
            if has_audio_preprocessor and has_image_preprocessor:
                required = "audio or image"
            elif has_audio_preprocessor:
                required = "audio"
            else:
                required = "image"
            msg = f"Model '{model}' requires {required} input for extraction"
            raise ValueError(msg)

        # Run media preprocessing in the shared bounded CPU pool.
        # This uses the shared preprocessing thread pool for better resource utilization
        # Pass instruction and task for models like Florence-2 that need them in the prompt
        task = options.get("task") if options else None
        prepared_batch = await preprocessor_registry.prepare(model, items, config, instruction=instruction, task=task)
        prepared_items = prepared_batch.items
        logger.info(
            "Preprocessed %d items for %s (total_cost=%d, prep_time_ms=%.1f)",
            len(prepared_items),
            model,
            prepared_batch.total_cost,
            timing.tokenization_ms or 0,
        )
    else:
        # Text/document model: cost is text characters or document byte size.
        # GLiNER/GLiClass tokenize internally; document adapters (Docling) parse internally.
        prepared_items = build_extract_prepared_items(items)

    timing.end_tokenization()

    # Start worker if not running
    worker = await registry.start_worker(model)

    # Submit to worker and await result
    future = await worker.submit_extract(
        prepared_items=prepared_items,
        items=items,
        labels=labels,
        output_schema=output_schema,
        instruction=instruction,
        options=options,
        timing=timing,
    )

    return await future


def _build_response(
    model: str,
    items: list,
    extraction_results: list[dict],
) -> ExtractResponse:
    """Build ExtractResponse from adapter extraction results.

    Args:
        model: Model name used for extraction.
        items: Original request items (for echoing IDs).
        extraction_results: Results from adapter.extract().

    Returns:
        ExtractResponse with results for each item.
    """
    results = []
    for i, result in enumerate(extraction_results):
        # Get item ID (echo from request or generate)
        item_id = items[i].id if items[i].id is not None else f"item-{i}"

        # Convert entity dicts to Entity objects
        entities = []
        for entity in result.get("entities", []):
            entities.append(
                Entity(
                    text=entity["text"],
                    label=entity["label"],
                    score=entity.get("score", 1.0),
                    start=entity.get("start"),
                    end=entity.get("end"),
                    bbox=entity.get("bbox"),
                )
            )

        # Convert relation dicts to Relation objects
        relations = []
        for rel in result.get("relations", []):
            relations.append(
                Relation(
                    head=rel["head"],
                    tail=rel["tail"],
                    relation=rel["relation"],
                    score=rel.get("score", 1.0),
                )
            )

        # Convert classification dicts to Classification objects
        classifications = []
        for cls in result.get("classifications", []):
            classifications.append(
                Classification(
                    label=cls["label"],
                    score=cls["score"],
                )
            )

        # Convert object dicts to DetectedObject objects
        objects = []
        for obj in result.get("objects", []):
            objects.append(
                DetectedObject(
                    label=obj["label"],
                    score=obj.get("score", 1.0),
                    bbox=obj["bbox"],
                )
            )

        item_result = ExtractResult(
            id=item_id,
            entities=entities,
            relations=relations,
            classifications=classifications,
            objects=objects,
            data=result.get("data", {}),
        )
        error = result.get("error")
        if isinstance(error, dict):
            item_result["error"] = ExtractItemErrorDetail(
                code=error["code"],
                message=error["message"],
            )
        results.append(item_result)

    return ExtractResponse(
        model=model,
        items=results,
    )


@router.post(
    "/extract/{model:path}",
    response_model=None,  # We handle serialization manually for content negotiation
    responses={
        200: {
            "description": "Extraction completed successfully",
            "model": ExtractResponseModel,
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
                    "schema": {"$ref": "#/components/schemas/ExtractRequestModel"},
                },
                "application/msgpack": {
                    "schema": {"$ref": "#/components/schemas/ExtractRequestModel"},
                },
            },
        },
    },
)
async def extract(
    model: str,
    http_request: Request,
    accept: Annotated[str | None, Header()] = None,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> MsgPackResponse | JSONResponse:
    """Extract entities or structured data from items.

    Supports both msgpack and JSON request bodies (Content-Type header).
    Returns msgpack by default, JSON if Accept header requests it.

    Args:
        model: Model name to use for extraction.
        http_request: FastAPI request object (for body and app state).
        accept: Accept header for response content negotiation.
        x_machine_profile: Machine profile header for routing validation.

    Returns:
        ExtractResponse with extraction results for each item.
        Format depends on Accept header: msgpack (default) or JSON.

    Raises:
        HTTPException: 400 for invalid input or profile mismatch, 404 if model not found,
            503 if not loaded.
    """
    # Validate machine profile header against worker identity (catches routing errors early)
    validate_machine_profile_header(x_machine_profile)

    # Start tracing span for extract operation
    with tracer.start_as_current_span("extract") as span:
        span.set_attribute("model", model)
        if x_machine_profile:
            span.set_attribute("machine_profile", x_machine_profile)

        request = await RequestParser.parse(http_request, ExtractRequest)

        # Set span attributes from request
        span.set_attribute("batch_size", len(request.items))

        registry = http_request.app.state.registry
        device = registry.device

        # Validate model state using helper
        model_checker = ModelStateChecker(registry, model, span)
        model_checker.check_exists()

        # Check model config supports extraction (extract-specific validation)
        config = registry.get_config(model)
        if config.tasks.extract is None:
            span.set_attribute("error", "unsupported_operation")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": f"Model '{model}' does not support extraction. "
                    f"Use an extraction model like GLiNER, GLiClass, or Florence-2.",
                },
            )

        # Continue model state validation
        model_checker.check_not_unloading()
        model_checker.check_not_loading()
        await model_checker.ensure_loaded(device)

        # Get params and resolve runtime options (outside inference try/except
        # so ValueError from invalid profiles returns 400, not 500)
        params = request.params
        labels = params.labels if params else None
        output_schema = params.output_schema if params else None
        instruction = params.instruction if params else None

        profile_name = (
            str(params.options.get("profile") or "default") if params is not None and params.options else "default"
        )
        options = resolve_runtime_options(config, params.options if params else None, span)

        # Request-level instruction takes precedence; fall back to profile instruction
        if instruction is None:
            instruction = options.get("instruction")

        items = request.items

        # Extract using worker with batching
        error_handler = InferenceErrorHandler(
            model,
            "extract",
            span,
            oom_retry_after_s=oom_retry_after_from_registry(registry),
            profile=profile_name,
            item_count=len(items),
        )
        try:
            worker_result = await _extract_via_worker(
                registry,
                model,
                items,
                labels=labels,
                output_schema=output_schema,
                instruction=instruction,
                options=options,
            )
            extract_output = cast("ExtractOutput", worker_result.output)
            extraction_results = ExtractHandler.format_output(extract_output)
            timing = worker_result.timing
        except QueueFullError as e:
            raise error_handler.handle_queue_full(e) from e
        except InputTooLongError as e:
            raise error_handler.handle_input_too_long(e) from e
        except ValueError as e:
            raise error_handler.handle_value_error(e) from e
        except Exception as e:
            raise error_handler.handle_inference_error(e, "Extraction") from e

        # Build response
        response = _build_response(model, items, extraction_results)

        if worker_telemetry_enabled():
            units: dict[str, int] = {}
            token_counts = extract_output.input_token_counts
            if (
                isinstance(token_counts, list)
                and len(token_counts) == len(items)
                and all(isinstance(count, int) and not isinstance(count, bool) for count in token_counts)
            ):
                units["input_tokens"] = sum(token_counts)
            pages = extract_output.pages
            if (
                isinstance(pages, list)
                and len(pages) == len(items)
                and all(isinstance(page_count, int) and not isinstance(page_count, bool) for page_count in pages)
            ):
                units["pages"] = sum(pages)
            worker_telemetry().item_completed(
                operation="extract",
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
