"""Does typological distance from English change watermark retention?

Single-hop round trip (en -> X -> en) through pivots spanning the typological
range, via the cloud LLM translator. For each pivot: mean detector z, detected
count, chrF++ (surface retention), and the retained/novel bigram green split.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from sie_sdk import SIEClient
from transformers import AutoConfig, AutoTokenizer, WatermarkingConfig

from wm_detector import WatermarkDetector
from transformers.generation.logits_process import WatermarkLogitsProcessor

import arms
from wm_common import HERE, SAMPLES_PATH

# Descriptive language labels, not a measured typological-distance scale.
PIVOTS = [
    ("<2de>", "German", "Germanic, close"),
    ("<2fr>", "French", "Romance, close"),
    ("<2ru>", "Russian", "Slavic, Cyrillic, medium"),
    ("<2ar>", "Arabic", "Semitic, RTL, far"),
    ("<2tr>", "Turkish", "Turkic, agglutinative, far"),
    ("<2ja>", "Japanese", "Japonic, SOV, no spaces, far"),
    ("<2zh>", "Chinese", "Sinitic, logographic, far"),
]

chrf = CHRF(word_order=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "distance-sweep-results.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists; choose a new --output so recorded "
            "sweep evidence is never replaced."
        )
    payload = json.loads(SAMPLES_PATH.read_text())
    params = payload["watermark_params"]
    tok = AutoTokenizer.from_pretrained(payload["wm_model_id"])
    config = AutoConfig.from_pretrained(payload["wm_model_id"])
    detector = WatermarkDetector(
        model_config=config, device="cpu",
        watermarking_config=WatermarkingConfig(**params), ignore_repeated_ngrams=True,
    )
    proc = WatermarkLogitsProcessor(vocab_size=config.vocab_size, device="cpu", **params)
    client = SIEClient(os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"),
                       timeout_s=600.0, api_key=os.environ["SIE_API_KEY"])
    arms.LANG_NAMES.update({t: n for t, n, _ in PIVOTS})

    watermarked = [s for s in payload["samples"] if s["kind"] == "watermarked"]

    def ids_of(text):
        return tok(text, add_special_tokens=False).input_ids

    def is_green(prev, cur):
        gl = proc._get_greenlist_ids(torch.tensor([[prev]]))
        flat = gl[0] if gl.dim() > 1 else gl
        return cur in set(flat.tolist())

    def detect_z(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        return float(detector(ids, return_dict=True).z_score[0])

    print(f"{'pivot':10s} {'typology':22s} {'meanz':>6s} {'det':>5s} {'chrf':>5s} "
          f"{'ret%':>5s} {'g|ret':>6s} {'g|nov':>6s}")
    all_rows = []
    for tag, name, note in PIVOTS:
        zs, chrfs = [], []
        ret_c, ret_g, nov_c, nov_g = 0, 0, 0, 0
        for s in watermarked:
            try:
                attacked = arms.round_trip(client, s["text"], [tag], translator="llm")
            except Exception as exc:
                print(f"  {name}: {s['id']} failed: {exc}")
                continue
            zs.append(detect_z(attacked))
            chrfs.append(chrf.sentence_score(attacked, [s["text"]]).score)
            orig_bg = set(zip(ids_of(s["text"])[:-1], ids_of(s["text"])[1:]))
            atk = ids_of(attacked)
            seen = set()
            for prev, cur in zip(atk[:-1], atk[1:]):
                if (prev, cur) in seen:
                    continue
                seen.add((prev, cur))
                g = int(is_green(prev, cur))
                if (prev, cur) in orig_bg:
                    ret_c += 1; ret_g += g
                else:
                    nov_c += 1; nov_g += g
            all_rows.append({"pivot": name, "id": s["id"], "z": zs[-1],
                             "chrf": chrfs[-1], "text": attacked})
        if not zs:
            print(f"{name}: no successful outputs")
            continue
        total = ret_c + nov_c
        print(f"{tag:10s} {note:22s} {statistics.mean(zs):6.2f} "
              f"{sum(1 for z in zs if z > 3.0):2d}/{len(zs):<2d} "
              f"{statistics.mean(chrfs):5.1f} {100*ret_c/total:5.1f} "
              f"{ret_g/max(ret_c,1):6.3f} {nov_g/max(nov_c,1):6.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(all_rows, indent=2, ensure_ascii=False))
    except FileExistsError:
        raise SystemExit(
            f"{args.output} already exists; choose a new --output so recorded "
            "sweep evidence is never replaced."
        )


if __name__ == "__main__":
    main()
