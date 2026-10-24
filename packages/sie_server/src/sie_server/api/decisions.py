"""OpenAI-compatible text decisions over SIE's typed decision models."""

from __future__ import annotations

import json
import math
import re
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sie_server.api import systemone
from sie_server.api.helpers import openai_error_response, read_bounded_request_body
from sie_server.api.routing import error_code, fallback_refusal, remote_routing
from sie_server.types.lora import LORA_SELECTION_FIELDS, InvalidLoraSelectionError, split_lora_selection

router = APIRouter(prefix="/v1", tags=["openai-compat"], dependencies=[Depends(remote_routing)])

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_QUESTIONS = 200
MAX_CHOICES = 255
MAX_MESSAGES = 4096
MAX_FIELD_CHARS = 1_048_576
MAX_SAFETY_IDENTIFIER_CHARS = 128
PROBABILITY_TOLERANCE = 1e-4


class DecisionRequest(BaseModel):
    model: str = Field(description="SIE decision model id, for example fastino/GLiNER2.5-Decide.")
    input: str | list[dict[str, Any]] = Field(description="Text or text-only messages, in conversation order.")
    questions: list[dict[str, Any]] = Field(
        min_length=1, max_length=MAX_QUESTIONS, description="Predicate, choice, or score questions, in answer order."
    )
    safety_identifier: str | None = Field(default=None, max_length=128)
    lora_adapter: str | None = Field(
        default=None,
        description="SIE extension: declared LoRA name or source; null selects base, omission keeps the profile default.",
    )
    lora: str | None = Field(
        default=None, description="Compatibility alias for lora_adapter; supplied aliases must agree."
    )
    lora_id: str | None = Field(
        default=None, description="Compatibility alias restricted to a declared source id; supplied aliases must agree."
    )


class DecisionUsage(BaseModel):
    input_tokens: int
    input_tokens_details: dict[str, int]
    output_tokens: int = 0
    output_tokens_details: dict[str, int]
    total_tokens: int


class DecisionResponse(BaseModel):
    model: str
    answers: list[dict[str, Any]]
    usage: DecisionUsage


def invalid(message: str, param: str | None = None, *, code: str = "invalid_request") -> HTTPException:
    return HTTPException(400, {"message": message, "type": "invalid_request_error", "param": param, "code": code})


def _text(value: Any, param: str) -> str:
    if not isinstance(value, str):
        raise invalid("must be a string", param)
    if len(value) > MAX_FIELD_CHARS:
        raise invalid(f"must contain at most {MAX_FIELD_CHARS} characters", param)
    return value


def _fields(value: dict[str, Any], allowed: set[str], param: str) -> None:
    for key in value:
        if key not in allowed:
            raise invalid("unsupported field", f"{param}.{key}" if param else key, code="unsupported_field")


def _input(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_MESSAGES:
        raise invalid(f"must be text or 1 to {MAX_MESSAGES} text messages", "input")
    for index, message in enumerate(value):
        param = f"input[{index}]"
        if not isinstance(message, dict):
            raise invalid("must be a message object", param)
        message = cast("dict[str, Any]", message)
        _fields(message, {"type", "role", "content", "phase"}, param)
        if message.get("type", "message") != "message":
            raise invalid("SIE decision models support text messages, not tool calls or media", f"{param}.type")
        role = message.get("role")
        if role not in ("user", "developer", "system", "assistant"):
            raise invalid("unsupported message role", f"{param}.role")
        if message.get("phase") not in (None, "commentary", "final_answer"):
            raise invalid("unsupported message phase", f"{param}.phase")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise invalid("must be text or a non-empty list of text parts", f"{param}.content")
        for part_index, part in enumerate(content):
            part_param = f"{param}.content[{part_index}]"
            if not isinstance(part, dict):
                raise invalid("must be a text part", part_param)
            part = cast("dict[str, Any]", part)
            kind = part.get("type")
            allowed = ("output_text", "refusal") if role == "assistant" else ("input_text",)
            if kind not in allowed:
                raise invalid("SIE decision models read text only; unsupported content part", f"{part_param}.type")
            field = "refusal" if kind == "refusal" else "text"
            _fields(part, {"type", field}, part_param)
            if not isinstance(part.get(field), str):
                raise invalid("must be a string", f"{part_param}.{field}")
    return cast("list[dict[str, Any]]", value)


def _description(value: str | bool, description: str | None, *, typed: bool = False) -> str:
    label = json.dumps(value, ensure_ascii=False) if typed else str(value)
    return f"{label}: {description}" if description else label


def normalize_request(body: Any) -> tuple[systemone.SystemOneRequest, list[dict[str, Any]]]:
    """Validate the public contract and preserve ordered, typed option identities.

    Raises:
        HTTPException: An unsupported or invalid request field.
    """
    if not isinstance(body, dict):
        raise invalid("request body must be a JSON object")
    _fields(body, {"model", "input", "questions", "safety_identifier", *LORA_SELECTION_FIELDS}, "")
    try:
        split_lora_selection(body)
    except InvalidLoraSelectionError as exc:
        raise invalid(str(exc), "lora_adapter") from exc
    model = _text(body.get("model"), "model")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", model) is None or ".." in model:
        raise invalid("invalid model id", "model")
    if body.get("safety_identifier") is not None:
        identifier = _text(body["safety_identifier"], "safety_identifier")
        if len(identifier) > MAX_SAFETY_IDENTIFIER_CHARS:
            raise invalid("must contain at most 128 characters", "safety_identifier")
    state = _input(body.get("input"))
    questions = body.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise invalid(f"must contain 1 to {MAX_QUESTIONS} questions", "questions")
    native: dict[str, Any] = {}
    for index, question in enumerate(questions):
        param = f"questions[{index}]"
        if not isinstance(question, dict):
            raise invalid("must be a question object", param)
        question = cast("dict[str, Any]", question)
        kind = question.get("type")
        if kind not in ("predicate", "choice", "score"):
            raise invalid("must be predicate, choice, or score", f"{param}.type")
        extra = {"choices"} if kind == "choice" else ({"levels"} if kind == "score" else set())
        _fields(question, {"type", "name", "instructions"} | extra, param)
        instructions = _text(question.get("instructions"), f"{param}.instructions")
        if "name" in question:
            _text(question["name"], f"{param}.name")
        normalized: dict[str, Any] = {"type": "noul" if kind == "predicate" else kind, "instructions": instructions}
        if kind in ("choice", "score"):
            field = "choices" if kind == "choice" else "levels"
            options = question.get(field)
            minimum, maximum = (2, MAX_CHOICES) if kind == "choice" else (1, systemone.MAX_SCORE_LEVELS)
            if not isinstance(options, list) or not minimum <= len(options) <= maximum:
                raise invalid(f"must contain {minimum} to {maximum} entries", f"{param}.{field}")
            descriptions: list[str] = []
            seen: set[tuple[type, str | bool]] = set()
            for option_index, option in enumerate(options):
                option_param = f"{param}.{field}[{option_index}]"
                if not isinstance(option, dict):
                    raise invalid("must be an object", option_param)
                option = cast("dict[str, Any]", option)
                value_field = "value" if kind == "choice" else "label"
                _fields(option, {value_field, "description"}, option_param)
                value = option.get(value_field)
                if kind == "choice" and type(value) not in (str, bool):
                    raise invalid("must be a string or boolean", f"{option_param}.value")
                if kind == "score" or isinstance(value, str):
                    _text(value, f"{option_param}.{value_field}")
                value = cast("str | bool", value)
                if kind == "choice":
                    identity = (type(value), value)
                    if identity in seen:
                        raise invalid("choice values must be unique", f"{option_param}.value")
                    seen.add(identity)
                description = (
                    _text(option["description"], f"{option_param}.description") if "description" in option else None
                )
                descriptions.append(_description(value, description, typed=kind == "choice"))
            normalized["criteria"] = (
                {f"option_{i}": description for i, description in enumerate(descriptions)}
                if kind == "choice"
                else descriptions
            )
        native[f"question_{index}"] = normalized
    return systemone.SystemOneRequest(
        model=model,
        state=state,
        questions=native,
        **{key: body[key] for key in LORA_SELECTION_FIELDS if key in body},
    ), questions


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid decision probability")
    return float(value)


def response_from_systemone(body: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Render answers without losing question order or boolean choice values.

    Raises:
        ValueError: The model returned an invalid answer distribution.
        KeyError: A required answer field is absent.
        TypeError: The model returned an invalid answer shape.
    """
    answers = []
    for index, question in enumerate(questions):
        answer = body["answers"][f"question_{index}"]
        kind = question["type"]
        output: dict[str, Any] = {"type": kind, "name": question.get("name")}
        if answer.get("type") == "refusal":
            output["type"] = "refusal"
        elif kind == "predicate":
            if answer.get("type") != "noul":
                raise ValueError("unexpected predicate answer type")
            output["probability"] = _probability(answer["noul"])
        else:
            if answer.get("type") != kind:
                raise ValueError("unexpected decision answer type")
            options = question["choices" if kind == "choice" else "levels"]
            keys = [f"option_{i}" if kind == "choice" else str(i) for i in range(len(options))]
            probabilities = answer["probabilities"]
            if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
                raise ValueError("decision probability keys do not match the options")
            values = [_probability(probabilities[key]) for key in keys]
            if abs(sum(values) - 1.0) > PROBABILITY_TOLERANCE:
                raise ValueError("decision probabilities must sum to one")
            if kind == "choice":
                choice = answer["choice"]
                if choice not in keys:
                    raise ValueError("decision choice is not an allowed option")
                output["choice"] = options[keys.index(choice)]["value"]
                output["probabilities"] = [
                    {"value": option["value"], "probability": p} for option, p in zip(options, values, strict=True)
                ]
                output["confidence"] = systemone.choice_confidence(values)
            else:
                output["score"] = sum(i * p for i, p in enumerate(values))
                output["probabilities"] = [
                    {"value": i, "label": option["label"], "probability": p}
                    for i, (option, p) in enumerate(zip(options, values, strict=True))
                ]
                output["confidence"] = systemone.score_confidence(values)
        answers.append(output)
    tokens = body["usage"]["input_tokens"]
    if type(tokens) is not int or tokens < 0:
        raise ValueError("invalid input token count")
    return {
        "model": body["model"],
        "answers": answers,
        "usage": {
            "input_tokens": tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": tokens,
        },
    }


def _translate_error(error: HTTPException) -> HTTPException:
    detail = error.detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        loc = first.get("loc", [])[1:]
        param = ".".join("input" if part == "state" else str(part) for part in loc) or None
        detail = {
            "message": first.get("msg", "invalid request"),
            "param": param,
            "code": first.get("code", "invalid_request"),
        }
    response_status = (
        status.HTTP_400_BAD_REQUEST if error.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT else error.status_code
    )
    return HTTPException(response_status, detail, headers=systemone._retry_headers(error.headers))


@router.post(
    "/decisions",
    response_model=DecisionResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": DecisionRequest.model_json_schema()}},
        }
    },
    responses={
        400: {"description": "Invalid or unsupported input"},
        404: {"description": "Model not found"},
        503: {"description": "Model loading or capacity exhausted"},
    },
)
async def decisions(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse:
    """Answer ordered predicate, choice, and score questions with a text decision model."""
    try:
        raw = await read_bounded_request_body(http_request, MAX_BODY_BYTES)
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise invalid("request body must be valid JSON") from exc
        request, questions = normalize_request(parsed)
        try:
            response = await systemone._system_one(request, http_request, x_machine_profile)
            body = response_from_systemone(json.loads(bytes(response.body)), questions)
        except (ArithmeticError, AttributeError, LookupError, TypeError, ValueError) as exc:
            raise HTTPException(
                500, {"message": "model returned an invalid decision result", "code": "INTERNAL_ERROR"}
            ) from exc
        headers = {
            key: value for key, value in response.headers.items() if key not in ("content-length", "content-type")
        }
        return JSONResponse(body, headers=headers)
    except HTTPException as exc:
        refusal = fallback_refusal(http_request, exc.status_code, error_code(exc))
        return openai_error_response(_translate_error(refusal or exc))
