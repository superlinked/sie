"""Native SIE runtime helpers for the contract-review agent."""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any

from sie_sdk import SIEAsyncClient

from .native_model import RequiredToolStep, SIENativeModel


def provision_timeout_from(cfg: dict[str, Any]) -> float:
    """Return the configured capacity-wait timeout."""
    return float(cfg["cluster"].get("provision_timeout_s", 900))


def model_for(
    model_id: str,
    client: SIEAsyncClient,
    *,
    stage: str,
    provision_timeout_s: float,
    required_tool_sequence: tuple[RequiredToolStep, ...] = (),
    api_calls: list[dict[str, Any]] | None = None,
) -> SIENativeModel:
    """Bind one SIE catalog model to the Agents SDK native model interface."""
    return SIENativeModel(
        model_id,
        client,
        stage=stage,
        provision_timeout_s=provision_timeout_s,
        required_tool_sequence=required_tool_sequence,
        api_calls=api_calls,
    )


@dataclass
class GenResult:
    """One native generation result plus observability fields."""

    text: str
    provision_s: float
    gen_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None

    @property
    def latency_s(self) -> float:
        return self.provision_s + self.gen_s

    @property
    def tokens_per_s(self) -> float | None:
        if self.completion_tokens and self.gen_s > 0:
            return self.completion_tokens / self.gen_s
        return None


@dataclass
class LedgerEntry:
    step: str
    model: str
    sie_fn: str
    warmup_s: float = 0.0
    latency_s: float = 0.0
    sent: str = ""
    got: str = ""
    throughput: str = ""


@dataclass
class Ledger:
    """Per-call observability for one agent run."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def record(
        self,
        step: str,
        model: str,
        sie_fn: str,
        *,
        warmup_s: float = 0.0,
        latency_s: float = 0.0,
        sent: str = "",
        got: str = "",
        throughput: str = "",
    ) -> None:
        self.entries.append(
            LedgerEntry(
                step,
                model,
                sie_fn,
                warmup_s,
                latency_s,
                sent,
                got,
                throughput,
            )
        )


@dataclass
class AppContext:
    """Shared dependencies handed to every tool through the Agents SDK."""

    sie: SIEAsyncClient
    cfg: dict[str, Any]
    ledger: Ledger
    contract_text: str
    scan_path: str
    db_path: str
    obligation_counterparty: str | None = None
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_agent: Any = None
    clause_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def provision_timeout_s(self) -> float:
        return provision_timeout_from(self.cfg)


def record_api_call(
    app: AppContext,
    sie_fn: str,
    requested_model: str,
    result: Any,
    *,
    stage: str,
) -> None:
    """Record only non-payload response metadata for checked run evidence."""
    rows = result if isinstance(result, list) else [result]
    response = next((row for row in rows if isinstance(row, dict)), {})
    request = response.get("request")
    request_row = request if isinstance(request, dict) else {}
    app.api_calls.append(
        {
            "stage": stage,
            "function": sie_fn,
            "requested_model": requested_model,
            "runtime_model": (
                response.get("model")
                if isinstance(response.get("model"), str)
                else None
            ),
            "request_id": (
                request_row.get("id")
                if isinstance(request_row.get("id"), str)
                else None
            ),
            "rate_book_version": request_row.get("rate_book_version"),
            "credits_debited": request_row.get("credits_debited"),
            "execution_identity_sha256": request_row.get("execution_identity_sha256"),
        }
    )


def _data_uri_image(value: str) -> tuple[bytes, str]:
    if not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("Native generation accepts inline image data, not remote URLs")
    header, encoded = value.split(",", 1)
    image_format = header.removeprefix("data:image/").split(";", 1)[0]
    return base64.b64decode(encoded, validate=True), image_format


def _instruction_prompt_and_images(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Build the role-labelled raw prompt accepted by native generate.

    Native text-only generation intentionally preserves raw-prompt execution.
    Image-bearing requests are rendered as one user turn by the server-side
    model chat template, as required by the native API contract.
    """
    sections: list[str] = []
    images: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if not isinstance(url, str):
                        raise ValueError("Image content part is missing image_url.url")
                    data, image_format = _data_uri_image(url)
                    images.append({"data": data, "format": image_format})
        if texts:
            sections.append(f"{role.upper()}\n" + "\n".join(texts))
    if not sections:
        raise ValueError("Generation messages contain no text")
    return "\n\n".join(sections), images


def _generation_result(result: dict[str, Any], elapsed_s: float) -> GenResult:
    usage = result.get("usage")
    usage_row = usage if isinstance(usage, dict) else {}
    request = result.get("request")
    request_row = request if isinstance(request, dict) else {}
    text = result.get("text")
    if not isinstance(text, str):
        raise TypeError("SIE native generate returned no text")
    prompt_tokens = usage_row.get("prompt_tokens")
    completion_tokens = usage_row.get("completion_tokens")
    request_id = request_row.get("id")
    return GenResult(
        text=text,
        provision_s=0.0,
        gen_s=elapsed_s,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=(
            completion_tokens if isinstance(completion_tokens, int) else None
        ),
        request_id=request_id if isinstance(request_id, str) else None,
    )


async def instruct_once(
    app: AppContext,
    model: str,
    messages: list[dict[str, Any]],
    *,
    stage: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: float | None = None,
    **extra: Any,
) -> GenResult:
    """Run a role-labelled instruction through native generate."""
    prompt, images = _instruction_prompt_and_images(messages)
    started = time.monotonic()
    call = app.sie.generate(
        model,
        prompt,
        max_new_tokens=max_tokens,
        images=images or None,
        temperature=temperature,
        wait_for_capacity=True,
        provision_timeout_s=(
            timeout_s if timeout_s is not None else app.provision_timeout_s
        ),
        **extra,
    )
    result = (
        await asyncio.wait_for(call, timeout=timeout_s)
        if timeout_s is not None
        else await call
    )
    record_api_call(app, "generate", model, result, stage=stage)
    return _generation_result(result, time.monotonic() - started)


async def prompt_once(
    app: AppContext,
    model: str,
    prompt: str,
    *,
    stage: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> GenResult:
    """Run a raw specialist prompt through the native generate primitive."""
    started = time.monotonic()
    result = await app.sie.generate(
        model,
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        wait_for_capacity=True,
        provision_timeout_s=app.provision_timeout_s,
    )
    record_api_call(app, "generate", model, result, stage=stage)
    return _generation_result(result, time.monotonic() - started)
