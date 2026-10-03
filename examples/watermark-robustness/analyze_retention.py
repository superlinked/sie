"""Decompose post-attack watermark signal into retained vs novel bigrams.

For every attacked text, split its scored (prev, cur) token bigrams by whether
the same bigram occurs in the original watermarked text. If the surviving
z-score is retention-driven, retained bigrams should be green at roughly the
original watermarked rate and novel bigrams at the 25% null.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import WatermarkLogitsProcessor

from wm_common import HERE, SAMPLES_PATH

# Loaded in main() from the --samples payload, so the analyzer always compares
# result rows against the source file that actually produced them.
payload = None
tok = None
proc = None


def ids_of(text):
    return tok(text, add_special_tokens=False).input_ids


def is_green(prev_id, cur_id):
    greenlist = proc._get_greenlist_ids(torch.tensor([[prev_id]]))
    flat = greenlist[0] if greenlist.dim() > 1 else greenlist
    return cur_id in set(flat.tolist())


def bigrams(ids):
    return list(zip(ids[:-1], ids[1:]))


def analyze(results_file):
    rows = json.loads((HERE / results_file).read_text())
    originals = {s["id"]: s["text"] for s in payload["samples"]}
    print(f"\n== {results_file}")
    print(f"{'arm':10s} {'n':>4s} {'retained%':>9s} {'green|ret':>9s} {'green|nov':>9s}")
    agg = {}
    for row in rows:
        kind = row.get("kind", next((s["kind"] for s in payload["samples"] if s["id"] == row["id"]), None))
        arm = row.get("arm", row.get("pivot", "unknown"))
        if kind != "watermarked" or arm == "identity":
            continue
        orig_bg = set(bigrams(ids_of(originals[row["id"]])))
        atk_ids = ids_of(row["text"])
        seen = set()
        counts = {"ret": [0, 0], "nov": [0, 0]}
        for prev, cur in bigrams(atk_ids):
            if (prev, cur) in seen:
                continue
            seen.add((prev, cur))
            bucket = "ret" if (prev, cur) in orig_bg else "nov"
            counts[bucket][0] += 1
            counts[bucket][1] += int(is_green(prev, cur))
        a = agg.setdefault(arm, {"ret": [0, 0], "nov": [0, 0]})
        for k in counts:
            a[k][0] += counts[k][0]
            a[k][1] += counts[k][1]
    for arm, a in agg.items():
        total = a["ret"][0] + a["nov"][0]
        ret_pct = 100 * a["ret"][0] / total
        g_ret = a["ret"][1] / max(a["ret"][0], 1)
        g_nov = a["nov"][1] / max(a["nov"][0], 1)
        print(f"{arm:10s} {total:4d} {ret_pct:8.1f}% {g_ret:9.3f} {g_nov:9.3f}")


def identity_check():
    """Sanity: our green computation should match the detector's ~0.6 on watermarked text."""
    total = green = 0
    for s in payload["samples"]:
        ids = ids_of(s["text"])
        seen = set()
        for prev, cur in bigrams(ids):
            if (prev, cur) in seen:
                continue
            seen.add((prev, cur))
            if s["kind"] == "watermarked":
                total += 1
                green += int(is_green(prev, cur))
    print(f"sanity: green fraction on watermarked identity = {green/total:.3f} (expect ~0.6)")


def main():
    global payload, tok, proc
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", default=["results.json"],
                        help="Result files to decompose (relative to this folder).")
    parser.add_argument("--samples", type=Path, default=SAMPLES_PATH,
                        help="Source file that produced the result rows; supplies "
                             "the model, watermark parameters, and original texts.")
    args = parser.parse_args()
    payload = json.loads(args.samples.read_text())
    params = payload["watermark_params"]
    tok = AutoTokenizer.from_pretrained(payload["wm_model_id"])
    config = AutoConfig.from_pretrained(payload["wm_model_id"])
    proc = WatermarkLogitsProcessor(vocab_size=config.vocab_size, device="cpu", **params)
    identity_check()
    for f in args.results:
        analyze(f)


if __name__ == "__main__":
    main()
