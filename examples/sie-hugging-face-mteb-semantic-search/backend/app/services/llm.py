import logging

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to backend/.env"
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def generate_text(prompt: str, max_tokens: int = 4096) -> str:
    """Send a prompt to OpenAI and return the assistant's response text."""
    client = _get_client()
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
