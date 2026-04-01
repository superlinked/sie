import logging
from typing import Annotated, Any, cast

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sie_sdk.types import DEFAULT_OUTPUT_DTYPE, DType, OutputDType, np_to_dtype

from sie_server.api.helpers import (
    InferenceErrorHandler,
    ModelStateChecker,
    RequestParser,
    ResponseBuilder,
    extract_request_context,
)
from sie_server.api.options import resolve_runtime_options
from sie_server.api.serialization import MsgPackResponse
from sie_server.api.validation import validate_machine_profile_header
from sie_server.config.model import ModelConfig
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.worker import QueueFullError
from sie_server.observability.metrics import record_request
from sie_server.observability.tracing import tracer
from sie_server.types.inputs import Item
from sie_server.types.openapi import EncodeResponseModel
from sie_server.types.outputs import DenseVector, EncodeResult, MultiVector, SparseVector
from sie_server.types.requests import EncodeRequest
from sie_server.types.responses import EncodeResponse, ErrorCode, TimingInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["encode"])


def _format_dense(
    embedding: np.ndarray,
    config: ModelConfig,
) -> DenseVector:
    """Format a dense embedding for response.

    Per DESIGN.md 13.2: Keep numpy arrays, don't convert to list.
    Quantization is now done by QuantizePostprocessor before this is called.
    """
    # For binary (packed uint8), dims represents the original dimension
    # For other dtypes, dims is just the array shape
    encode_task = config.tasks.encode
    dense_dim = encode_task.dense.dim if encode_task and encode_task.dense else None
    is_binary = embedding.dtype == np.uint8 and dense_dim and embedding.shape[0] < dense_dim
    if is_binary:
        original_dims: int = dense_dim  # type: ignore[assignment]
        dtype: DType = "binary"
    else:
        original_dims = dense_dim if dense_dim is not None else embedding.shape[0]
        dtype = np_to_dtype(embedding)

    return DenseVector(
        dims=original_dims,
        dtype=dtype,
        values=embedding,
    )


def _format_sparse(
    result: dict[str, Any],
    config: ModelConfig,
) -> SparseVector:
    """Format a sparse embedding for response.

    Per DESIGN.md 13.2: Keep numpy arrays, don't convert to list.
    Quantization is now done by QuantizePostprocessor before this is called.

    Note: Sparse vectors only support float32 and float16 dtypes.
    Quantization doesn't make sense for sparse (indices dominate storage).
    """
    indices = result["indices"]
    values = result["values"]

    # Ensure numpy arrays
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    encode_task = config.tasks.encode
    sparse_dim = encode_task.sparse.dim if encode_task and encode_task.sparse else None
    return SparseVector(
        dims=sparse_dim,
        dtype=np_to_dtype(values),
        indices=indices,
        values=values,
    )


def _format_multivector(
    embeddings: np.ndarray,
    config: ModelConfig,
) -> MultiVector:
    """Format a multivector embedding for response.

    Per DESIGN.md 13.2: Keep numpy arrays, don't convert to list.
    Quantization is now done by QuantizePostprocessor before this is called.

    Note: Quantization is applied per-token. ColBERTv2 uses 2-bit quantization
    with good results, so int8/binary are well-defined for multivector.
    """
    # For binary (packed uint8), token_dims represents the original dimension
    encode_task = config.tasks.encode
    mv_dim = encode_task.multivector.dim if encode_task and encode_task.multivector else None
    is_binary = embeddings.dtype == np.uint8 and mv_dim and embeddings.shape[1] < mv_dim
    if is_binary:
        original_token_dims: int = mv_dim  # type: ignore[assignment]
        dtype: DType = "binary"
    else:
        original_token_dims = mv_dim if mv_dim is not None else embeddings.shape[1]
        dtype = np_to_dtype(embeddings)

    return MultiVector(
        token_dims=original_token_dims,
        num_tokens=embeddings.shape[0],
        dtype=dtype,
        values=embeddings,
    )


def _build_response_items(
    items: list[Item],
    results: list[dict[str, Any]],
    config: ModelConfig,
) -> list[EncodeResult]:
    """Build response items from adapter results.

    Args:
        items: Input items (for preserving IDs).
        results: Raw adapter results (already quantized by postprocessor).
        config: Model configuration (for dims).

    Returns:
        List of EncodeResult items with formatted embeddings.
    """
    response_items = []
    for i, item in enumerate(items):
        result_dict: dict[str, Any] = {}
        item_id = item.id
        if item_id is not None:
            result_dict["id"] = item_id

        # Add outputs from adapter results, keeping numpy arrays
        adapter_result = results[i]
        if "dense" in adapter_result:
            result_dict["dense"] = _format_dense(adapter_result["dense"], config)
        if "sparse" in adapter_result:
            result_dict["sparse"] = _format_sparse(adapter_result["sparse"], config)
        if "multivector" in adapter_result:
            result_dict["multivector"] = _format_multivector(adapter_result["multivector"], config)

        response_items.append(EncodeResult(**result_dict))

    return response_items


@router.post(
    "/encode/{model:path}",
    response_model=None,  # We handle serialization manually for content negotiation
    responses={
        200: {
            "description": "Embeddings generated successfully",
            "model": EncodeResponseModel,
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
                    "schema": {"$ref": "#/components/schemas/EncodeRequestModel"},
                },
                "application/msgpack": {
                    "schema": {"$ref": "#/components/schemas/EncodeRequestModel"},
                },
            },
        },
    },
)
async def encode(
    model: str,
    http_request: Request,
    accept: Annotated[str | None, Header()] = None,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> MsgPackResponse | JSONResponse:
    """Generate embeddings for input items.

    Supports both msgpack and JSON request bodies (Content-Type header).
    Returns msgpack by default, JSON if Accept header requests it.

    Args:
        model: Model name to use for encoding.
        http_request: FastAPI request object (for body and app state).
        accept: Accept header for response content negotiation.
        x_machine_profile: Machine profile header for routing validation.

    Returns:
        EncodeResponse with embeddings for each input item.
        Format depends on Accept header: msgpack (default) or JSON.

    Raises:
        HTTPException: 400 for invalid input or profile mismatch, 404 if model not found,
            503 if not loaded.
    """
    # Validate machine profile header against worker identity (catches routing errors early)
    validate_machine_profile_header(x_machine_profile)

    # Start tracing span for encode operation
    with tracer.start_as_current_span("encode") as span:
        span.set_attribute("model", model)
        if x_machine_profile:
            span.set_attribute("machine_profile", x_machine_profile)

        request = await RequestParser.parse(http_request, EncodeRequest)

        # Set span attributes from request
        params = request.params
        span.set_attribute("batch_size", len(request.items))
        if params:
            span.set_attribute("output_types", ",".join(params.output_types or ["dense"]))

        registry = http_request.app.state.registry
        device = registry.device

        # Extract request context for structured logging
        ctx = extract_request_context(http_request, model, registry)

        # Validate model state using helper
        model_checker = ModelStateChecker(registry, model, span)
        model_checker.check_exists()
        model_checker.check_not_unloading()
        model_checker.check_not_loading()
        await model_checker.ensure_loaded(device)

        # Get config
        config = registry.get_config(model)

        # Get instruction from params
        instruction = params.instruction if params else None

        # Resolve profile and merge runtime options
        request_options = params.options if params else None
        profile_name = request_options.get("profile") if request_options else None
        options = resolve_runtime_options(config, request_options, span)

        # Extract is_query from options (moved from top-level param to options)
        is_query = bool(options.get("is_query", False))

        # Get instruction: request param > profile > None
        # Request-level instruction takes precedence to allow per-request overrides
        # instruction was already extracted from params above
        if instruction is None:
            instruction = options.get("instruction")

        # Check if LoRA is specified and ensure it's loaded
        lora = options.get("lora_id")
        if lora is not None:
            try:
                is_ready, is_loading = await registry.ensure_lora_loaded_async(model, lora)
                if is_loading:
                    # LoRA is loading - return 503 with retry hint
                    span.set_attribute("error", "lora_loading")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "code": ErrorCode.LORA_LOADING.value,
                            "message": f"LoRA '{lora}' is loading for model '{model}', please retry",
                        },
                        headers={"Retry-After": "1"},
                    )
                if not is_ready:
                    # LoRA load failed
                    span.set_attribute("error", "lora_load_failed")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail={
                            "code": ErrorCode.INFERENCE_ERROR.value,
                            "message": f"Failed to load LoRA '{lora}' for model '{model}'",
                        },
                    )
            except ValueError as e:
                # Model doesn't support LoRA
                span.set_attribute("error", "lora_not_supported")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": ErrorCode.INVALID_INPUT.value,
                        "message": str(e),
                    },
                ) from e

        # Get output_types: profile > request param > default
        output_types: list[str] = options.get("output_types") or (params.output_types if params else None) or ["dense"]

        # Get output_dtype: request param > profile > default
        # Request param takes precedence to allow per-request overrides
        output_dtype: OutputDType = cast(
            "OutputDType",
            (params.output_dtype if params else None) or options.get("output_dtype") or DEFAULT_OUTPUT_DTYPE,
        )

        # Add resolved output_dtype to options for postprocessor registry
        options["output_dtype"] = output_dtype

        # Validate output types against model capabilities + profile-enabled outputs
        supported_outputs = set(config.outputs)
        # Profiles can enable additional output types via postprocessors (e.g., muvera enables dense)
        if profile_name and profile_name in config.profiles:
            profile_output_types = options.get("output_types")
            if profile_output_types:
                supported_outputs = supported_outputs | set(profile_output_types)
        requested_outputs = set(output_types)
        unsupported = requested_outputs - supported_outputs
        if unsupported:
            span.set_attribute("error", "unsupported_output_types")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": f"Model '{model}' does not support output types: {unsupported}. "
                    f"Supported: {supported_outputs}",
                },
            )

        # Translate output types for postprocessor-enabled outputs
        # MUVERA produces 'dense' by postprocessing 'multivector', so we request
        # 'multivector' from adapter and let postprocessor add 'dense'
        muvera_enabled = options.get("muvera") is not None
        needs_translation = muvera_enabled and "dense" in output_types
        if needs_translation:
            # Build adapter types: remove 'dense', ensure 'multivector'
            adapter_output_types = [t for t in output_types if t != "dense"]
            if "multivector" not in adapter_output_types:
                adapter_output_types.append("multivector")
        else:
            adapter_output_types = output_types  # No copy needed - not mutating

        items = request.items

        # Run encoding (preprocess → execute)
        error_handler = InferenceErrorHandler(model, "encode", span, ctx=ctx)
        try:
            results, timing = await EncodePipeline.run_encode(
                registry=registry,
                model=model,
                items=items,
                output_types=adapter_output_types,
                instruction=instruction,
                config=config,
                is_query=is_query,
                options=options,
            )
        except QueueFullError as e:
            raise error_handler.handle_queue_full(e) from e
        except ValueError as e:
            raise error_handler.handle_value_error(e) from e
        except Exception as e:
            raise error_handler.handle_inference_error(e) from e

        # Build response (quantization already done by postprocessor)
        response_items = _build_response_items(items, results, config)

        # Get timing info (if timing was tracked)
        timing_info: TimingInfo | None = None
        if timing is not None:
            timing.finish()
            timing_info = TimingInfo(
                total_ms=timing.total_ms,
                queue_ms=timing.queue_ms,
                tokenization_ms=timing.tokenization_ms,
                inference_ms=timing.inference_ms,
                postprocessing_ms=timing.postprocessing_ms if timing.postprocessing_ms > 0 else None,
            )
            # Add timing info to span
            span.set_attribute("timing.total_ms", timing.total_ms)
            span.set_attribute("timing.tokenize_ms", timing.tokenization_ms)
            span.set_attribute("timing.queue_ms", timing.queue_ms)
            span.set_attribute("timing.inference_ms", timing.inference_ms)
            if timing.postprocessing_ms > 0:
                span.set_attribute("timing.postprocessing_ms", timing.postprocessing_ms)

        response = EncodeResponse(model=model, items=response_items, timing=timing_info)

        # Record metrics for successful request
        record_request(
            model=model,
            endpoint="encode",
            status="success",
            timing=timing,
            request_id=ctx.request_id,
            api_key=ctx.api_key,
            queue_depth=ctx.queue_depth,
        )

        # Build response headers and return
        headers = ResponseBuilder.build_headers(timing)
        return ResponseBuilder.build_response(response, accept, headers, convert_for_json=True)
