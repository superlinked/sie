"""Validate OpenAI raw completion events using the shared choice/usage parser."""

from __future__ import annotations

import json
from typing import Any, cast

from sie_server.adapters._generation_base import FinishReason, GenerationChunk
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote._openai_chat import ChatStreamParser


def _invalid() -> RemoteUpstreamError:
    return RemoteUpstreamError("upstream returned an invalid completion response")


def _logprobs(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _invalid()
    tokens, probabilities = value.get("tokens"), value.get("token_logprobs")
    alternatives = value.get("top_logprobs")
    if not isinstance(tokens, list) or not isinstance(probabilities, list) or len(tokens) != len(probabilities):
        raise _invalid()
    if alternatives is not None and (not isinstance(alternatives, list) or len(alternatives) != len(tokens)):
        raise _invalid()
    content = []
    for index, (token, probability) in enumerate(zip(tokens, probabilities, strict=True)):
        top = None if alternatives is None else alternatives[index]
        if top is not None and not isinstance(top, dict):
            raise _invalid()
        content.append(
            {
                "token": token,
                "logprob": probability,
                "bytes": None,
                "top_logprobs": [{"token": key, "logprob": prob} for key, prob in (top or {}).items()],
            }
        )
    return {"content": content}


class CompletionStreamParser:
    """A single raw prompt produces text deltas, then one terminal with exact usage."""

    def __init__(self, *, logprobs: bool = False) -> None:
        self._logprobs_requested = logprobs
        self._chat = ChatStreamParser("upstream")
        self._finish: str | None = None
        self._usage: dict[str, Any] | None = None
        self._seen_text = False

    def parse(self, data: bytes) -> GenerationChunk | None:
        if data == b"[DONE]":
            self._chat.parse(data)
            assert self._usage is not None
            return GenerationChunk(
                text_delta="",
                done=True,
                finish_reason="error" if self._finish == "content_filter" else cast("FinishReason", self._finish),
                prompt_tokens=self._usage["prompt_tokens"],
                completion_tokens=self._usage["completion_tokens"],
                cached_tokens=self._usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                error_code="inference_error" if self._finish == "content_filter" else None,
                error_message="the upstream refused the output" if self._finish == "content_filter" else None,
            )
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError):
            raise _invalid() from None
        if not isinstance(payload, dict) or "error" in payload or not isinstance(payload.get("choices"), list):
            raise _invalid()
        choices = []
        for choice in payload["choices"]:
            if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
                raise _invalid()
            reason = choice.get("finish_reason")
            if reason not in (None, "stop", "length", "content_filter"):
                raise _invalid()
            clean = {
                "index": choice.get("index"),
                "delta": {"content": choice["text"]},
                "finish_reason": reason,
            }
            has_reasoning = any(choice.get(key) not in (None, "") for key in ("reasoning_content", "reasoning"))
            if self._logprobs_requested and choice["text"] and choice.get("logprobs") is None and not has_reasoning:
                raise _invalid()
            if "logprobs" in choice:
                clean["logprobs"] = None if has_reasoning else _logprobs(choice["logprobs"])
            choices.append(clean)
        event = self._chat.parse(json.dumps({"choices": choices, "usage": payload.get("usage")}).encode())
        assert event is not None
        if "usage" in event:
            self._usage = event["usage"]
        if not event["choices"]:
            return None
        choice = event["choices"][0]
        if choice["finish_reason"] is not None:
            self._finish = choice["finish_reason"]
        text = choice["delta"]["content"]
        first = bool(text) and not self._seen_text
        self._seen_text |= bool(text)
        raw_logprobs = choice.get("logprobs")
        return GenerationChunk(
            text_delta=text,
            is_first=first,
            logprobs=None
            if raw_logprobs is None or not self._logprobs_requested
            else tuple(raw_logprobs.get("content") or []),
        )

    def finish(self) -> None:
        self._chat.finish()
