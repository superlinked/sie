#!/usr/bin/env python3
"""Reproduce the /ocr figures from the recorded run. No API key, no network.

    python3 fetch.py
    python3 score.py                 # the recorded run, checked against the page
    python3 score.py --rescore runs  # your own run.py output, scored the same way

The recorded run: SIE's hosted LightOnOCR-2-1B, GPT-5.4 mini, and PaddleOCR, Tesseract and
EasyOCR self-hosted, on the same 600 photos (172 GNHK handwritten notes, 100 CORD receipt
photos, 328 SROIE receipt scans). The score is exact-word recall: the share of the words
on each photo that an output holds exactly. This verifies every file against the manifest
pinned below, recomputes each arm's recall from the per-image counts, prints them beside
each arm's price per 1,000 images, and exits non-zero if any figure superlinked.com/ocr
publishes does not come out.

`--rescore DIR` instead scores the text in DIR/outputs.jsonl (what run.py writes) against
the reference words in DIR/references.jsonl, with the same token rule. Standard library
only in both modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

# The SHA-256 of manifest.json at the dataset revision fetch.py pins. It lives here,
# outside the evidence, because a digest inside a file cannot authenticate that file.
MANIFEST_SHA256 = "74cade75f456b9d2d5ac34e09b94ccf82c0ef4b0d17c4a59cb6ffc6ef23e9d5b"

ARMS = {
    "sie-lightonocr-2-1b": "SIE LightOnOCR-2-1B",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "paddleocr-pp-ocrv5": "PaddleOCR PP-OCRv5, self-hosted",
    "tesseract-5": "Tesseract 5, self-hosted",
    "easyocr": "EasyOCR, self-hosted",
    "sie-glm-ocr": "GLM-OCR (not on SIE Cloud)",
    "sie-paddleocr-vl-1.5": "PaddleOCR-VL-1.5 (not on SIE Cloud)",
}
SETS = ("gnhk", "cord", "sroie")

# What superlinked.com/ocr publishes: words read correctly in whole percent, ours rounded
# down and every rival up, and $ per 1,000 images (GPT-5.4 mini at its real-time price on
# its recorded tokens; self-hosted engines at $0 plus your servers).
PAGE = {
    "percent": {
        "sie-lightonocr-2-1b": 92,
        "gpt-5.4-mini": 92,
        "paddleocr-pp-ocrv5": 82,
        "tesseract-5": 57,
        "easyocr": 56,
    },
    "usd_per_1k": {"sie-lightonocr-2-1b": 1.16, "gpt-5.4-mini": 2.36},
}

# The token rule, fixed before the run: NFKC; markup stripped from model output; split on
# whitespace; leading and trailing punctuation and a trailing full stop dropped. SROIE's
# reference text is upper-case, so SROIE is compared case-insensitively.
_EDGE = "\"'`()[]{}<>,;:!?"


def _strip_markup(text: str) -> str:
    import re

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*[-:| ]+\s*$", " ", text, flags=re.MULTILINE)
    return re.sub(r"[|#*_`]|\$+", " ", text)


def tokens(text: str, *, markup: bool) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    if markup:
        text = _strip_markup(text)
    out = []
    for raw in text.split():
        token = raw.strip(_EDGE)
        if token.endswith("."):
            token = token[:-1].strip(_EDGE)
        if token:
            out.append(token)
    return out


def counts(reference: list[str], output: str, *, caseless: bool) -> tuple[int, int]:
    ref = [t for word in reference for t in tokens(word, markup=False)]
    hyp = tokens(output, markup=True)
    if caseless:
        ref, hyp = [t.casefold() for t in ref], [t.casefold() for t in hyp]
    have = Counter(hyp)
    read = 0
    for token in ref:
        if have[token] > 0:
            have[token] -= 1
            read += 1
    return len(ref), read


def verified() -> dict:
    manifest_bytes = (EVIDENCE / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise SystemExit("evidence/manifest.json is not the recorded run's manifest; run python3 fetch.py")
    manifest = json.loads(manifest_bytes)
    for relative, digest in manifest["files"].items():
        path = EVIDENCE / relative
        if relative.startswith("outputs/") and not path.exists():
            continue  # fetched only with --outputs
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"evidence/{relative} does not match the manifest")
    return manifest


def recorded() -> int:
    verified()
    results = json.loads((EVIDENCE / "results.json").read_text())
    per_image = json.loads((EVIDENCE / "per_image.json").read_text())
    gpt = results["gpt_5_4_mini"]
    rate = gpt["usd_per_1m"]
    # Real-time against real-time: GPT-5.4 mini's standard token price, not its batch tier.
    gpt_usd = (gpt["mean_input_tokens"] * rate["input"] + gpt["mean_output_tokens"] * rate["output"]) / 1e6 * 1000
    prices = {"sie-lightonocr-2-1b": results["sie"]["price_page_usd_per_1k"], "gpt-5.4-mini": gpt_usd}

    ok = True
    print(f"{'Arm':38} {'Words':>7} {'Digits':>7}  {'GNHK':>6} {'CORD':>6} {'SROIE':>6}  $/1k images")
    for arm, name in ARMS.items():
        rows = per_image["arms"][arm]
        if len(rows) != sum(results["images"].values()):
            raise SystemExit(f"{arm}: {len(rows)} images, expected {sum(results['images'].values())}")
        n, read, values, values_read = (sum(row[k] for row in rows) for k in (1, 2, 3, 4))
        recall, value_recall = 100 * read / n, 100 * values_read / values
        by_set = {}
        for s in SETS:
            mine = [row for row in rows if row[0].startswith(f"{s}/")]
            by_set[s] = 100 * sum(r[2] for r in mine) / sum(r[1] for r in mine)
        if abs(recall / 100 - results["arms"][arm]["word_recall"]) > 1e-12:
            print(f"  {arm}: recall does not re-derive", file=sys.stderr)
            ok = False
        price = prices.get(arm, 0.0 if not arm.startswith("sie-") else None)
        shown = "" if price is None else f"${price:.2f}"
        print(
            f"{name:38} {recall:7.2f} {value_recall:7.2f}  "
            f"{by_set['gnhk']:6.1f} {by_set['cord']:6.1f} {by_set['sroie']:6.1f}  {shown}"
        )
        if arm in PAGE["percent"]:
            rounded = math.floor(recall) if arm.startswith("sie-") else math.ceil(recall)
            if rounded != PAGE["percent"][arm]:
                print(f"  the page says {PAGE['percent'][arm]}% for {name}; the run gives {rounded}%", file=sys.stderr)
                ok = False
    for arm, usd in PAGE["usd_per_1k"].items():
        if abs(round(prices[arm], 2) - usd) > 1e-9:
            print(f"  the page prices {ARMS[arm]} at ${usd:.2f}; the run gives ${prices[arm]:.4f}", file=sys.stderr)
            ok = False

    tests = results["paired_tests"]["rivals"]
    print("\nLightOnOCR-2-1B minus each arm, points of word recall, 95% interval (paired image bootstrap):")
    for arm, t in tests.items():
        verdict = "worse" if t["worse_than"] else "not shown as worse"
        print(f"  {ARMS[arm]:36} {t['diff_points']:+6.2f}  ({t['ci95'][0]:+.2f} to {t['ci95'][1]:+.2f})  {verdict}")
    print(
        "\nNot run (no credentials): "
        + ", ".join(results["not_run"])
        + ". The page makes no accuracy claim about them."
    )
    print("\nEvery published figure reproduces." if ok else "\nA published figure did not reproduce.")
    return 0 if ok else 1


def rescore(run_dir: Path) -> int:
    references = {row["id"]: row for row in map(json.loads, (run_dir / "references.jsonl").read_text().splitlines())}
    outputs = {row["id"]: row for row in map(json.loads, (run_dir / "outputs.jsonl").read_text().splitlines())}
    missing, unexpected = references.keys() - outputs.keys(), outputs.keys() - references.keys()
    if missing or unexpected or not references:
        print(
            f"The run is incomplete: {len(missing)} photos have no output, {len(unexpected)} outputs have no reference."
            " Re-run run.py.",
            file=sys.stderr,
        )
        return 1
    totals: dict[str, list[int]] = {}
    for image_id, out in outputs.items():
        ref = references[image_id]
        n, read = counts(ref["words"], out.get("text") or "", caseless=ref["set"] == "sroie")
        total = totals.setdefault(ref["set"], [0, 0, 0])
        total[0] += n
        total[1] += read
        total[2] += 1
    for s, (n, read, images) in sorted(totals.items()):
        print(f"{s}: {images} images, {100 * read / n:.1f}% of {n} words read exactly")
    n = sum(t[0] for t in totals.values())
    read = sum(t[1] for t in totals.values())
    print(f"all: {100 * read / n:.1f}% of {n} words")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", type=Path, help="score run.py's output in this directory instead")
    args = parser.parse_args()
    return rescore(args.rescore) if args.rescore else recorded()


if __name__ == "__main__":
    sys.exit(main())
