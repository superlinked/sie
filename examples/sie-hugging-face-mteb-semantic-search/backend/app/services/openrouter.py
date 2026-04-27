import asyncio
import logging
from typing import Optional

from openai import APIStatusError, AsyncOpenAI, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None

_RETRY_MAX = 5
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env"
            )
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )
    return _client


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env"
            )
        _async_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )
    return _async_client


def _log_usage(response) -> None:
    text = response.choices[0].message.content or ""
    usage = response.usage
    if usage:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details else None
        logger.info(
            "OpenRouter usage: prompt_tokens=%s, completion_tokens=%s, "
            "reasoning_tokens=%s, content_chars=%d, finish_reason=%s",
            usage.prompt_tokens,
            usage.completion_tokens,
            reasoning,
            len(text),
            response.choices[0].finish_reason,
        )
    else:
        logger.info("OpenRouter response: %d chars", len(text))


def generate_text(
    prompt: str,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> str:
    """Send a prompt to OpenRouter and return the assistant's response text."""
    client = _get_client()
    model_name = model or settings.openrouter_model

    logger.info(
        "Calling OpenRouter model=%s (prompt length: %d chars, max_tokens: %d)",
        model_name,
        len(prompt),
        max_tokens,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    _log_usage(response)
    return (response.choices[0].message.content or "").strip()


async def generate_text_async(
    prompt: str,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Async version of generate_text with automatic retry on 429 rate-limit."""
    client = _get_async_client()
    model_name = model or settings.openrouter_model

    async with semaphore or asyncio.Semaphore(1):
        for attempt in range(_RETRY_MAX):
            logger.info(
                "Calling OpenRouter (async) model=%s (prompt length: %d chars, max_tokens: %d)",
                model_name,
                len(prompt),
                max_tokens,
            )
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.4,
                )
                _log_usage(response)
                return (response.choices[0].message.content or "").strip()
            except APIStatusError as exc:
                if exc.status_code == 429 and attempt < _RETRY_MAX - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Rate-limited (429); retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        _RETRY_MAX,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    raise RuntimeError("Unreachable: retry loop exited without return or raise")
