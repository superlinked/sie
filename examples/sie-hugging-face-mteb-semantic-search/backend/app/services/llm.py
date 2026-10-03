import asyncio
import logging
from typing import Optional

from openai import OpenAI

from app.config import settings
from app.services import orcarouter, openrouter

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to backend/.env"
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _generate_openai(prompt: str, max_tokens: int = 4096) -> str:
    """Send a prompt to OpenAI and return the assistant's response text."""
    client = _get_openai_client()
    logger.info(
        "Calling OpenAI %s (prompt length: %d chars, max_tokens: %d)",
        settings.openai_model,
        len(prompt),
        max_tokens,
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    text = response.choices[0].message.content or ""
    logger.info("OpenAI response: %d chars", len(text))
    return text.strip()


def generate_text(
    prompt: str,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> str:
    """Generate text with the configured provider (defaults to OpenRouter).

    Supported providers (LLM_PROVIDER): ``openrouter`` (default),
    ``orcarouter``, and ``openai``.
    """
    provider = settings.llm_provider.strip().lower()
    if provider == "orcarouter":
        return orcarouter.generate_text(prompt, max_tokens=max_tokens, model=model)
    if provider == "openai":
        return _generate_openai(prompt, max_tokens=max_tokens)
    return openrouter.generate_text(prompt, max_tokens=max_tokens, model=model)


async def generate_text_async(
    prompt: str,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Async generate_text with the configured provider (defaults to OpenRouter).

    Supported providers (LLM_PROVIDER): ``openrouter`` (default),
    ``orcarouter``, and ``openai`` (sync fallback).
    """
    provider = settings.llm_provider.strip().lower()
    if provider == "orcarouter":
        return await orcarouter.generate_text_async(
            prompt, max_tokens=max_tokens, model=model, semaphore=semaphore
        )
    if provider == "openai":
        return await asyncio.to_thread(_generate_openai, prompt, max_tokens)
    return await openrouter.generate_text_async(
        prompt, max_tokens=max_tokens, model=model, semaphore=semaphore
    )
