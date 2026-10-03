"""Paraphrase transformations for the watermark robustness experiment.

The paraphraser changes wording and structure using sampled decoding.
The experiment does not isolate whether sampling randomness causes the
observed difference: deterministic rewriting can also erase a signal.
Arms: single paraphrase, two passes, and paraphrase plus an Arabic round trip.

Runs only on our own watermarked samples; detection needs our key. This
measures decay of a watermark we own, the standard robustness protocol.
"""

from arms import LLM_TRANSLATOR_MODEL, round_trip
from sie_chat import chat_text

PARAPHRASE_MODEL = LLM_TRANSLATOR_MODEL
_PROMPT = (
    "Paraphrase the following text. Preserve the meaning exactly, but rewrite it "
    "in completely different words and a different sentence structure. Do not copy "
    "phrases from the original. Output only the paraphrase, nothing else.\n\n{text}"
)


def paraphrase_llm(client, text: str, temperature: float = 1.0, seed: int = 0) -> str:
    return chat_text(client, {
        "model": PARAPHRASE_MODEL,
        "messages": [{"role": "user", "content": _PROMPT.format(text=text)}],
        "max_tokens": 1024,
        "temperature": temperature,
        "top_p": 0.95,
        "seed": seed,
    })


def paraphrase_recursive(client, text: str, rounds: int = 2) -> str:
    current = text
    for i in range(rounds):
        current = paraphrase_llm(client, current, seed=i)
    return current


def paraphrase_then_translate(client, text: str) -> str:
    return round_trip(client, paraphrase_llm(client, text), ["<2ar>"], translator="llm")
