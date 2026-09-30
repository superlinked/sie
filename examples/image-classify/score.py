#!/usr/bin/env python3
"""Reproduce the /image-classify figures from the recorded run. No API key, no network, no inference spend.

    python3 fetch.py
    python3 score.py

The task: fill in three catalogue fields (type, colour, material) for 938 held-out product photos, each field chosen
from the caller's own list. The primary figure is the share of photos with all three fields right.

Everything is re-derived here, not read back:

* SIE arms: every photo-to-prompt cosine is recorded; this script picks each field's prompt variant on the dev set,
  then answers each test photo with the highest-cosine value.
* Vision LLM arms: the recorded JSON answers, parsed here; an answer outside the lists counts as wrong.
* AWS Rekognition: the recorded DetectLabels responses, mapped to the caller's lists with the map that was fitted on
  dev (specific labels before their parents, so "Ottoman" outranks "Furniture").

It then checks every figure against page-evidence.json, the file superlinked.com/image-classify renders, and exits
non-zero on any difference.

Standard library only.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

EVIDENCE = Path(__file__).resolve().parent / "evidence"
FIELDS = ("type", "colour", "material")

# The label prompts the SIE arms encode: three templates per field and their mean.
TEMPLATES = {
    "type": ("a photo of {a} {v}.", "{v}", "a product photo of {a} {v}."),
    "colour": ("a photo of a {v} product.", "{v}", "a {v} object."),
    "material": ("a photo of a product made of {v}.", "{v}", "a photo of a {v} product."),
}
VARIANTS = ("t0", "t1", "t2", "ensemble")

# Real-time prices for one million photos a month (read 2026-09-30).
SIE_USD_PER_1K_IMAGES = 0.0232  # google/siglip-so400m-patch14-384 on SIE Cloud
LLM_USD_PER_1M = {  # real-time list price (input, output) per million tokens
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "claude-haiku-4-5": (1.00, 5.00),
}
REKOGNITION_USD_PER_1K = 1.00 + 0.75  # DetectLabels + image properties (needed for colour), first-million tier
# Each LLM is charted at its better resolution and priced at the cheaper one within a point of it.
LLMS = {
    "GPT-6 Luna": ("gpt-6-luna@1024", "gpt-6-luna@1024"),
    "GPT-5.4 mini": ("gpt-5.4-mini@1024", "gpt-5.4-mini@512"),
    "Claude Haiku 4.5": ("claude-haiku-4-5@1024", "claude-haiku-4-5@1024"),
    "GPT-5.4 nano": ("gpt-5.4-nano@512", "gpt-5.4-nano@512"),
}


def load(name: str) -> Any:
    path = EVIDENCE / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    if name.endswith(".gz"):
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def render(template: str, value: str) -> str:
    return template.format(a="an" if value[:1].lower() in "aeiou" else "a", v=value)


def sie_answers(arm: str, vocab: dict[str, list[str]], dev: list[dict]) -> dict[str, dict[str, str]]:
    report = load(f"scores/{arm}.json.gz")
    index = {p: k for k, p in enumerate(report["prompts"])}
    rows = dict(zip(report["image_ids"], report["cosine"], strict=True))

    def answer(row: list[float], field: str, variant: str) -> str:
        best, best_score = "", -math.inf
        for value in vocab[field]:
            if variant == "ensemble":
                s = sum(row[index[render(t, value)]] for t in TEMPLATES[field]) / 3
            else:
                s = row[index[render(TEMPLATES[field][int(variant[1])], value)]]
            if s > best_score:
                best, best_score = value, s
        return best

    chosen = {}
    for field in FIELDS:
        hits = {v: sum(answer(rows[r["image_id"]], field, v) == r[field] for r in dev) for v in VARIANTS}
        chosen[field] = max(VARIANTS, key=lambda v: (hits[v], -VARIANTS.index(v)))
    return {image_id: {f: answer(row, f, chosen[f]) for f in FIELDS} for image_id, row in rows.items()}


def llm_answers(name: str, vocab: dict[str, list[str]]) -> tuple[dict[str, dict[str, str | None]], float]:
    rows = load(f"answers/{name}.json.gz")
    out = {}
    for r in rows:
        try:
            answer = json.loads(r["text"])
        except (KeyError, ValueError):
            answer = {}
        out[r["image_id"]] = {f: answer.get(f) if answer.get(f) in vocab[f] else None for f in FIELDS}
    price_in, price_out = LLM_USD_PER_1M[name.split("@")[0]]
    tin = statistics.mean(r["tokens_in"] for r in rows)
    tout = statistics.mean(r["tokens_out"] for r in rows)
    return out, tin * price_in + tout * price_out  # real time, per million photos


def rekognition_answers(maps: dict[str, dict[str, str]]) -> dict[str, dict[str, str | None]]:
    out = {}
    for r in load("answers/rekognition.json.gz"):
        raw = r["raw"]
        labels = raw.get("Labels", [])
        parents = {p["Name"] for label in labels for p in label.get("Parents", [])}
        ordered = [
            label["Name"]
            for _, label in sorted(
                enumerate(labels), key=lambda kv: (kv[1]["Name"] in parents, -kv[1]["Confidence"], kv[0])
            )
        ]
        props = raw.get("ImageProperties", {})
        colours = [
            c["SimplifiedColor"]
            for c in sorted(props.get("Foreground", {}).get("DominantColors", []), key=lambda c: -c["PixelPercent"])
        ] + [c["SimplifiedColor"] for c in sorted(props.get("DominantColors", []), key=lambda c: -c["PixelPercent"])]
        candidates = {"type": ordered, "material": ordered, "colour": colours}
        out[r["image_id"]] = {f: next((maps[f][n] for n in candidates[f] if n in maps[f]), None) for f in FIELDS}
    return out


def pct(answers: dict[str, dict[str, Any]], test: list[dict], key: str) -> float:
    if key == "all":
        hits = sum(all(answers[r["image_id"]][f] == r[f] for f in FIELDS) for r in test)
    else:
        hits = sum(answers[r["image_id"]][key] == r[key] for r in test)
    return round(100 * hits / len(test), 2)


def main() -> int:
    sets = load("sets.json")
    vocab, dev, test = sets["vocabulary"], sets["dev"], sets["test"]
    page = load("page-evidence.json")
    rows: list[tuple[str, str, dict[str, float], float]] = []

    ours = sie_answers("sie-siglip-so400m-patch14-384", vocab, dev)
    rows.append(
        (
            "page_model",
            "SIE SigLIP so400m-384",
            {k: pct(ours, test, k) for k in ("all", *FIELDS)},
            math.ceil(SIE_USD_PER_1K_IMAGES * 1000),
        )
    )  # ours rounded up
    nxt = sie_answers("sie-siglip-so400m-patch14-224", vocab, dev)
    rows.append(("next_model", "SIE SigLIP so400m-224", {k: pct(nxt, test, k) for k in ("all", *FIELDS)}, math.nan))
    rek = rekognition_answers(load("maps/rekognition-amendment1.json"))
    rows.append(
        (
            "rekognition",
            "AWS Rekognition",
            {k: pct(rek, test, k) for k in ("all", *FIELDS)},
            REKOGNITION_USD_PER_1K * 1000,
        )
    )
    for name, (scored, priced) in LLMS.items():
        answers, _ = llm_answers(scored, vocab)
        _, usd = llm_answers(priced, vocab)
        rows.append((scored, name, {k: pct(answers, test, k) for k in ("all", *FIELDS)}, usd))

    print(f"{len(test)} test photos: share with the field right (%), and $ per million photos at real-time prices\n")
    print(f"{'':24s} {'all three':>9s} {'type':>6s} {'colour':>6s} {'material':>8s} {'$ / 1M':>8s}")
    for _, name, figures, usd in sorted(rows, key=lambda r: -r[2]["all"]):
        cost = "" if math.isnan(usd) else f"{usd:8.0f}"
        print(
            f"{name:24s} {figures['all']:9.1f} {figures['type']:6.1f} {figures['colour']:6.1f} {figures['material']:8.1f} {cost}"
        )

    published = {"page_model": page["page_model"], "next_model": page["next_model"], "rekognition": page["rekognition"]}
    published |= {llm["id"]: llm for llm in page["llms"]}
    bad = 0
    for key, name, figures, usd in rows:
        for k, v in figures.items():
            if abs(published[key][k] - v) > 0.005:
                print(f"MISMATCH {name} {k}: re-derived {v}, published {published[key][k]}", file=sys.stderr)
                bad += 1
        if "price" in published[key] and "usd_per_photo" in published[key]["price"]:
            pub = published[key]["price"]["usd_per_photo"] * 1e6
            if abs(pub - usd) > 0.5:
                print(f"MISMATCH {name} price: re-derived ${usd:.1f}, published ${pub:.1f}", file=sys.stderr)
                bad += 1
    if bad:
        return 1
    print("\nEvery figure matches page-evidence.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
