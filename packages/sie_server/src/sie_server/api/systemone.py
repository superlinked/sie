"""System One-compatible typed decisions: ``POST /v1/systemone``.

A client written for the System One API (the TypeSafe SDKs, or anything that
sends the same JSON) points its base URL at SIE and keeps its code. The route
answers one ``state`` against a mapping of typed ``questions`` (``noul``,
``choice``, ``score``) with an SIE decision model, through the same worker path
as ``POST /v1/extract/{model}``:

- request ``{model, state, questions}``; the model is named in the body, and
  fields the route does not read are ignored;
- response ``{model, answers, usage: {input_tokens, output_tokens}}``, always
  JSON;
- a question may omit ``instructions`` or send ``null``;
- ``choice`` and ``score`` confidences follow the System One API's published
  definitions (see :func:`choice_confidence` and :func:`score_confidence`);
- a state or question set that does not fit the model's row is refused with
  422 instead of being truncated (``extract``'s default);
- validation errors are 422 in FastAPI's ``{"detail": [...]}`` shape, other
  errors ``{"detail": {"error_type", "code", "message"}}``, and a retryable
  error's ``Retry-After`` is at most 60 seconds, with ``retry-after-ms``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from sie_server.adapters.errors import InputTooLongError
from sie_server.api.extract import _extract_via_worker
from sie_server.api.helpers import (
    InferenceErrorHandler,
    ModelStateChecker,
    ResponseBuilder,
    check_sdk_version,
    oom_retry_after_from_registry,
    validated_total,
)
from sie_server.api.options import resolve_runtime_options
from sie_server.api.routing import error_code, fallback_refusal, remote_routing, route_request
from sie_server.api.validation import validate_machine_profile_header
from sie_server.core.worker import QueueFullError
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.observability.tracing import tracer
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled
from sie_server.types.inputs import Item, item_size_error
from sie_server.types.lora import LORA_SELECTION_FIELDS
from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig
    from sie_server.core.inference_output import ExtractOutput

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["systemone"], dependencies=[Depends(remote_routing)])

# Adapters that answer typed questions (``output_schema`` question mappings).
DECISION_ADAPTERS = frozenset(
    {
        "sie_server.adapters.gliner2.decide:GLiNER2DecideAdapter",
        "sie_server.adapters.gliner2.decide_lora:GLiNER2DecideLoraAdapter",
        "sie_server.adapters.laya.adapter:LayaAdapter",
        "sie_server.adapters.laya.runtime_lora:LayaLoraAdapter",
        "sie_server.adapters.pplx_decider.adapter:PerplexityDecisionAdapter",
        "sie_server.adapters.pplx_decider.runtime_lora:PerplexityDecisionLoraAdapter",
    }
)

# Score levels per question, as the System One API accepts.
MAX_SCORE_LEVELS = 10

# Longest Retry-After the route sends. Clients built for the System One API
# (and LiteLLM) wait for at most 60 seconds before falling back.
MAX_RETRY_AFTER_S = 60

# Request fields for media the decision models cannot read.
_MEDIA_FIELDS = ("images", "audio", "videos")

_ERROR_TYPES = {
    status.HTTP_400_BAD_REQUEST: "invalid_request_error",
    status.HTTP_401_UNAUTHORIZED: "authentication_error",
    status.HTTP_403_FORBIDDEN: "permission_error",
    status.HTTP_404_NOT_FOUND: "not_found_error",
    status.HTTP_413_CONTENT_TOO_LARGE: "invalid_request_error",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "invalid_request_error",
    status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_error",
}


class SystemOneRequest(BaseModel):
    """A System One request: one state and its typed questions."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        ..., min_length=1, description="SIE model id of a decision model, e.g. fastino/GLiNER2.5-Decide."
    )
    state: str | dict[str, Any] | list[Any] = Field(
        ..., description="The text to judge, or a JSON object, or a list of conversation turns."
    )
    questions: dict[str, Any] = Field(
        ...,
        min_length=1,
        description='Typed questions by id: {"type": "noul" | "choice" | "score", "instructions"?, "criteria"?}.',
    )


class SystemOneUsage(BaseModel):
    """Tokens the model read. Decision models generate no tokens."""

    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    """Answers by question id, in the System One response shape."""

    model: str
    answers: dict[str, dict[str, Any]]
    usage: SystemOneUsage


def choice_confidence(probabilities: list[float]) -> float:
    """System One choice confidence: ``(p_max - 1/n) / (1 - 1/n)``, within [0, 1].

    0 when every option is equally likely, 1 when one option has all the
    probability. A single option is certain.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    floor = 1.0 / n
    return min(1.0, max(0.0, (max(probabilities) - floor) / (1.0 - floor)))


def score_confidence(probabilities: list[float]) -> float:
    """System One score confidence: ``max(0, 1 - spread / even)``.

    ``spread`` is the expected distance in levels from the most likely level,
    ``sum(p_i * |i - m|)``, and ``even`` the expected distance of a uniform
    distribution from the middle level, ``sum(|i - (n - 1) / 2|) / n``. A
    single level is certain.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    mode = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return min(1.0, max(0.0, 1.0 - spread / even))


def _invalid(message: str, loc: list[str | int]) -> HTTPException:
    """A 422 in FastAPI's validation-error shape, which System One clients parse."""
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=[{"type": "value_error", "loc": ["body", *loc], "msg": message}],
    )


def _is_decision_model(config: ModelConfig) -> bool:
    return config.tasks.extract is not None and any(
        profile.adapter_path in DECISION_ADAPTERS for profile in config.profiles.values()
    )


def normalize_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    """The questions as the decision adapters read them.

    A missing or ``null`` ``instructions`` becomes ``""``. A score question
    with a list of levels must have 1 to 10 of them, none ``null``. Everything else is left to
    the adapter's own validation, which rejects what it cannot answer.

    Raises:
        HTTPException: 422 naming the question.
    """
    normalized: dict[str, Any] = {}
    for qid, question in questions.items():
        if not isinstance(question, Mapping):
            raise _invalid(f"question {qid!r} must be an object", ["questions", qid])
        question = dict(question)
        if question.get("instructions") is None:
            question["instructions"] = ""
        if question.get("type") == "score":
            levels = question.get("criteria")
            if isinstance(levels, list):
                if not levels:
                    raise _invalid(f"score question {qid!r} has no levels", ["questions", qid, "criteria"])
                if len(levels) > MAX_SCORE_LEVELS:
                    raise _invalid(
                        f"score question {qid!r} has {len(levels)} levels; at most {MAX_SCORE_LEVELS} are allowed",
                        ["questions", qid, "criteria"],
                    )
                if any(level is None for level in levels):
                    raise _invalid(f"score question {qid!r} has a null level", ["questions", qid, "criteria"])
        normalized[qid] = question
    return normalized


def to_answer(answer: Mapping[str, Any]) -> dict[str, Any]:
    """One adapter answer in the System One shape, with System One confidences.

    Raises:
        ValueError: A probability is not a number, or a score level key is not an integer.
    """
    result = dict(answer)
    kind = result.get("type")
    probabilities = result.get("probabilities")
    if kind == "choice" and isinstance(probabilities, Mapping):
        result["confidence"] = choice_confidence([float(p) for p in probabilities.values()])
    elif kind == "score" and isinstance(probabilities, Mapping):
        ordered = [p for _, p in sorted((int(key), float(p)) for key, p in probabilities.items())]
        result["confidence"] = score_confidence(ordered)
    return result


def _error_message(detail: Any) -> str:
    if isinstance(detail, Mapping):
        inner = detail.get("error")
        if isinstance(inner, Mapping) and isinstance(inner.get("message"), str):
            return inner["message"]
        if isinstance(detail.get("message"), str):
            return detail["message"]
    if isinstance(detail, str):
        return detail
    return "request failed"


def _error_code(detail: Any) -> str | None:
    if isinstance(detail, Mapping):
        code = detail.get("code")
        if isinstance(code, str):
            return code
        inner = detail.get("error")
        if isinstance(inner, Mapping) and isinstance(inner.get("code"), str):
            return inner["code"]
    return None


def _retry_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """The error's headers, with ``Retry-After`` capped and mirrored as ``retry-after-ms``."""
    out = dict(headers or {})
    retry_key = next((key for key in out if key.lower() == "retry-after"), None)
    if retry_key is None:
        return out
    try:
        seconds = float(out[retry_key])
    except ValueError:
        return out
    if not math.isfinite(seconds) or seconds < 0:
        return out
    del out[retry_key]
    seconds = min(seconds, MAX_RETRY_AFTER_S)
    out["Retry-After"] = str(max(1, math.ceil(seconds)))
    out["retry-after-ms"] = str(int(seconds * 1000))
    return out


def error_response(error: HTTPException) -> JSONResponse:
    """An SIE error in the System One error shapes.

    Any 400 (for example ``INVALID_INPUT`` or ``INPUT_TOO_LONG``) becomes 422 in
    FastAPI's validation shape, as System One answers a request it cannot
    validate. Every other status keeps its code, with
    ``{"detail": {"error_type", "code", "message"}}``.
    """
    headers = _retry_headers(error.headers)
    if isinstance(error.detail, list):
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail}, headers=headers)
    message = _error_message(error.detail)
    code = _error_code(error.detail)
    if error.status_code == status.HTTP_400_BAD_REQUEST:
        entry: dict[str, Any] = {"type": "value_error", "loc": ["body"], "msg": message}
        if code is not None:
            entry["code"] = code
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": [entry]}, headers=headers
        )
    error_type = _ERROR_TYPES.get(error.status_code, "api_error")
    detail: dict[str, Any] = {"error_type": error_type, "message": message}
    if code is not None:
        detail["code"] = code
    return JSONResponse(status_code=error.status_code, content={"detail": detail}, headers=headers)


def _item_error(result: Mapping[str, Any]) -> HTTPException | None:
    """The 422 for an item the adapter refused (a state that does not fit, an unreadable state).

    The entry carries the adapter's code (``INPUT_TOO_LONG``, ``INVALID_INPUT``).
    """
    error = result.get("error")
    if not isinstance(error, Mapping):
        return None
    code = _error_code(error) or ErrorCode.INVALID_INPUT.value
    message = str(error.get("message", "the state could not be read"))
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=[{"type": "value_error", "loc": ["body", "state"], "msg": message, "code": code}],
    )


@router.post(
    "/systemone",
    response_model=SystemOneResponse,
    responses={
        200: {"description": "Every question answered"},
        404: {"description": "Model not found"},
        422: {
            "description": (
                "Invalid request: a malformed question, a model that does not answer typed questions, "
                "or a state or question set that does not fit the model's window"
            )
        },
        503: {"description": "Model loading or server busy; retry after Retry-After (at most 60 seconds)"},
    },
)
async def system_one(
    request: SystemOneRequest,
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse:
    """Answer typed questions about one state (System One-compatible).

    Clients written for the System One API work by changing their base URL.
    """
    try:
        return await _system_one(request, http_request, x_machine_profile)
    except HTTPException as error:
        return error_response(fallback_refusal(http_request, error.status_code, error_code(error)) or error)
    except (ArithmeticError, AttributeError, LookupError, TypeError, ValueError):
        logger.exception("Model returned an invalid System One result")
        return error_response(
            HTTPException(
                500, {"message": "model returned an invalid decision result", "code": ErrorCode.INTERNAL_ERROR.value}
            )
        )


async def _system_one(
    request: SystemOneRequest,
    http_request: Request,
    x_machine_profile: str | None,
) -> JSONResponse:
    validate_machine_profile_header(x_machine_profile)
    check_sdk_version(http_request)
    model = request.model

    with tracer.start_as_current_span("systemone") as span:
        span.set_attribute("model", model)
        registry = http_request.app.state.registry
        ModelStateChecker(registry, model, span).check_exists()

        config = registry.get_config(model)
        if not _is_decision_model(config):
            span.set_attribute("error", "unsupported_operation")
            raise _invalid(
                f"Model '{model}' does not answer typed questions. Use a decision model such as "
                "fastino/GLiNER2.5-Decide or convaiinnovations/laya-typed-decisions.",
                ["model"],
            )

        extra = request.model_extra or {}
        for field in _MEDIA_FIELDS:
            if extra.get(field) is not None:
                raise _invalid(f"'{field}' is not supported: SIE decision models read text only", [field])

        questions = normalize_questions(request.questions)
        state = request.state
        item = Item(text=state) if isinstance(state, str) else Item(metadata={"state": state})
        if size_error := item_size_error(item, "state"):
            raise _invalid(size_error, ["state"])

        request_options = {
            "overflow_policy": "error",
            **{key: extra[key] for key in LORA_SELECTION_FIELDS if key in extra},
        }
        options = resolve_runtime_options(config, request_options, span)
        route = await route_request(http_request, model, span, queued_items=1, request_options=request_options)
        error_handler = InferenceErrorHandler(
            model,
            "extract",
            span,
            oom_retry_after_s=oom_retry_after_from_registry(registry),
            profile="default",
            item_count=1,
        )
        try:
            worker_result = await _extract_via_worker(
                registry, route.key, [item], output_schema=questions, options=options
            )
        except QueueFullError as e:
            raise error_handler.handle_queue_full(e) from e
        except InputTooLongError as e:
            raise error_handler.handle_input_too_long(e) from e
        except ValueError as e:
            raise error_handler.handle_value_error(e) from e
        except Exception as e:
            raise error_handler.handle_inference_error(e, "Decision") from e

        output = cast("ExtractOutput", worker_result.output)
        result = ExtractHandler.format_output(output)[0]
        if refusal := _item_error(result):
            span.set_attribute("error", _error_code(result.get("error")) or ErrorCode.INVALID_INPUT.value)
            raise refusal

        data = result.get("data") or {}
        try:
            answers = {qid: to_answer(data[qid]) for qid in questions if isinstance(data.get(qid), Mapping)}
        except (TypeError, ValueError) as e:
            raise error_handler.handle_inference_error(RuntimeError(f"unreadable answer: {e}"), "Decision") from e
        if len(answers) != len(questions):
            missing = sorted(set(questions) - set(answers))
            raise error_handler.handle_inference_error(
                RuntimeError(f"model returned no answer for questions {missing}"), "Decision"
            )

        counted_tokens = validated_total(output.input_token_counts, 1)
        if counted_tokens is None:
            logger.warning("Model %s reported no input token count for a System One request", model)
        input_tokens = counted_tokens or 0
        timing = worker_result.timing
        if worker_telemetry_enabled():
            worker_telemetry().item_completed(
                operation="extract",
                outcome="success",
                model=model,
                profile="default",
                duration_s=timing.total_ms / 1000.0,
                item_count=1,
                tokenization_s=timing.tokenization_ms / 1000.0,
                inference_s=timing.inference_ms / 1000.0,
                postprocessing_s=timing.postprocessing_ms / 1000.0,
                units={} if counted_tokens is None else {"input_tokens": counted_tokens},
            )

        body = SystemOneResponse(model=model, answers=answers, usage=SystemOneUsage(input_tokens=input_tokens))
        headers = ResponseBuilder.build_headers(timing)
        headers.update(route.headers())
        return JSONResponse(content=body.model_dump(), headers=headers)
