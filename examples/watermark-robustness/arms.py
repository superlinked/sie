"""Attack arms: SIE round-trip translation, with and without NER protection.

Local-server translation goes sentence-by-sentence through MADLAD-400-3B
with greedy decoding. The NER arm
uses GLiNER through SIE's extract primitive to shield entity spans behind
placeholders across the round trip, then restores them.
"""

import os
import re

from sie_sdk import SIEClient
from sie_chat import chat_text

from wm_common import NER_LABELS, NER_MODEL, TRANSLATION_MODEL

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z؀-ۿÀ-ſ“\"'])")
PLACEHOLDER = "⟦{}⟧"  # ⟦i⟧
_PLACEHOLDER_RE = re.compile(r"⟦\s*(\d+)\s*⟧")

LLM_TRANSLATOR_MODEL = os.environ.get("SIE_GENERATOR_MODEL", "Qwen/Qwen3.8-27B-FP8")
LANG_NAMES = {"<2ar>": "Arabic", "<2hu>": "Hungarian", "<2en>": "English"}


def sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def translate(client: SIEClient, text: str, target_tag: str) -> str:
    out = []
    for sent in sentences(text):
        result = client.generate(
            TRANSLATION_MODEL,
            f"{target_tag} {sent}",
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        out.append(result["text"].strip())
    return " ".join(out)


def translate_llm(client: SIEClient, text: str, target_tag: str) -> str:
    language = LANG_NAMES[target_tag]
    prompt = (
        f"Translate the following text to {language}. "
        f"Output only the {language} translation, nothing else.\n\n{text}"
    )
    return chat_text(client, {
        "model": LLM_TRANSLATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    })


def round_trip(client: SIEClient, text: str, hops: list[str], translator: str = "madlad") -> str:
    """hops are intermediate MADLAD tags, e.g. ["<2ar>"] or ["<2ar>", "<2hu>"]."""
    step = translate if translator == "madlad" else translate_llm
    current = text
    for tag in hops:
        current = step(client, current, tag)
    return step(client, current, "<2en>")


def extract_entities(client: SIEClient, text: str) -> list[dict]:
    result = client.extract(NER_MODEL, {"text": text}, labels=NER_LABELS)
    entities = result.get("entities", [])
    spans = [e for e in entities if e.get("start") is not None and e.get("end") is not None]
    spans.sort(key=lambda e: e["start"])
    merged = []
    for e in spans:
        if merged and e["start"] < merged[-1]["end"]:
            continue
        merged.append(e)
    return merged


def protect(text: str, entities: list[dict]) -> tuple[str, dict[int, str]]:
    mapping = {}
    out = text
    for i, e in enumerate(reversed(entities)):
        idx = len(entities) - 1 - i
        mapping[idx] = text[e["start"]:e["end"]]
        out = out[: e["start"]] + PLACEHOLDER.format(idx) + out[e["end"]:]
    return out, mapping


def restore(text: str, mapping: dict[int, str]) -> tuple[str, int, int]:
    """Returns (restored_text, placeholders_survived, placeholders_total)."""
    survived = set()

    def _sub(m: re.Match) -> str:
        idx = int(m.group(1))
        if idx in mapping:
            survived.add(idx)
            return mapping[idx]
        return m.group(0)

    restored = _PLACEHOLDER_RE.sub(_sub, text)
    return restored, len(survived), len(mapping)


def ner_protected_round_trip(
    client: SIEClient, text: str, hops: list[str], translator: str = "madlad"
) -> tuple[str, dict]:
    entities = extract_entities(client, text)
    masked, mapping = protect(text, entities)
    tripped = round_trip(client, masked, hops, translator)
    restored, survived, total = restore(tripped, mapping)
    stats = {"entities": total, "placeholders_survived": survived}
    return restored, stats
