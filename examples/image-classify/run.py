#!/usr/bin/env python3
"""Tag one product photo with your own lists of types, colours and materials, live on SIE Cloud.

    export SIE_API_KEY=sk-sie-...
    uv run run.py photo.jpg
    uv run run.py photo.jpg --types "sofa, armchair, ottoman" --colours "blue, grey" --materials "velvet, leather"

One encode call carries the photo and every label prompt; each field takes the value whose prompt has the highest
cosine with the photo. The prompts are the ones the recorded study picked on its dev set for this model: the mean of
three templates for type, "a {colour} object." for colour and "a photo of a product made of {material}." for
material. With no lists given, the study's own lists are used (python3 fetch.py first).

Nothing here is needed to reproduce the published figures; score.py does that from the recorded run with no key.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sie_sdk import SIEClient

MODEL = "google/siglip-so400m-patch14-384"
TYPE_TEMPLATES = ("a photo of {a} {v}.", "{v}", "a product photo of {a} {v}.")
COLOUR_TEMPLATE = "a {v} object."
MATERIAL_TEMPLATE = "a photo of a product made of {v}."


def render(template: str, value: str) -> str:
    return template.format(a="an" if value[:1].lower() in "aeiou" else "a", v=value)


def parse(text: str | None) -> list[str] | None:
    return [v.strip() for v in text.split(",") if v.strip()] if text else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("--types")
    parser.add_argument("--colours")
    parser.add_argument("--materials")
    args = parser.parse_args()

    lists = {"type": parse(args.types), "colour": parse(args.colours), "material": parse(args.materials)}
    if not all(lists.values()):
        sets = Path(__file__).resolve().parent / "evidence" / "sets.json"
        if not sets.exists():
            raise SystemExit("Give --types, --colours and --materials, or run python3 fetch.py for the study's lists")
        vocab = json.loads(sets.read_text())["vocabulary"]
        lists = {field: values or vocab[field] for field, values in lists.items()}

    prompts: dict[str, list[list[str]]] = {
        "type": [[render(t, v) for t in TYPE_TEMPLATES] for v in lists["type"]],
        "colour": [[render(COLOUR_TEMPLATE, v)] for v in lists["colour"]],
        "material": [[render(MATERIAL_TEMPLATE, v)] for v in lists["material"]],
    }
    texts = [p for field in prompts.values() for group in field for p in group]

    client = SIEClient(api_key=os.environ["SIE_API_KEY"], base_url="https://api.superlinked.com")
    image = {"data": args.photo.read_bytes(), "format": "png" if args.photo.suffix.lower() == ".png" else "jpeg"}
    vecs = client.encode(MODEL, [{"images": [image]}, *[{"text": t} for t in texts]])
    mat = np.array([v["dense"] for v in vecs], dtype=np.float64)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    cosine = dict(zip(texts, (mat[1:] @ mat[0]).tolist(), strict=True))

    for field, groups in prompts.items():
        scores = [float(np.mean([cosine[p] for p in group])) for group in groups]
        best = int(np.argmax(scores))
        print(f"{field:9s} {lists[field][best]}")


if __name__ == "__main__":
    main()
