"""One hosted generation request; no automatic retries of billable work."""

import httpx
from sie_sdk.client.errors import SIEError


def chat_text(client, payload):
    try:
        response = client.chat_completions(
            **payload, wait_for_capacity=False, max_oom_retries=0,
        )
    except (httpx.RequestError, SIEError) as exc:
        raise RuntimeError(
            f"SIE request failed ({type(exc).__name__}). Check access, balance and usage before retrying."
        ) from None
    try:
        choice = response["choices"][0]
        text = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("SIE returned an unexpected response; no retry was made.") from None
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("SIE did not finish normally; do not score a truncated or refused rewrite.")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("SIE returned no translation/paraphrase text.")
    return text.strip()
