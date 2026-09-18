"""Measure watermark decay under paraphrase attacks vs the translation baseline."""

import argparse
import getpass
import json
import os
import statistics
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer, util
from sie_sdk import SIEClient
from transformers import AutoConfig, AutoTokenizer, WatermarkingConfig

from wm_detector import WatermarkDetector
from transformers.generation.logits_process import WatermarkLogitsProcessor

from arms import LLM_TRANSLATOR_MODEL, round_trip
from paraphrase_arms import paraphrase_llm, paraphrase_recursive, paraphrase_then_translate
from wm_common import HERE, SAMPLES_PATH

chrf = CHRF(word_order=2)
Z = 3.0
ARM_NAMES = ["rt_ar", "paraphrase_1x", "paraphrase_2x", "paraphrase+rt_ar"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES_PATH)
    parser.add_argument("--output", type=Path, default=HERE / "runs/sie-demo.json")
    parser.add_argument("--limit", type=int, default=1, help="Number of source passages (default: 1).")
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=["rt_ar", "paraphrase_1x"])
    parser.add_argument("--ask-key", action="store_true", help="Read a key without echo or saving it.")
    parser.add_argument("--sie-url", default=os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"))
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive.")
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output; saved evidence is never overwritten.")
    if not args.samples.is_file():
        parser.error("--samples must point to a saved source file.")
    return args


def main() -> None:
    args = parse_args()
    api_key = getpass.getpass("SIE API key (hidden): ") if args.ask_key else os.environ.get("SIE_API_KEY")
    if args.sie_url.rstrip("/") == "https://api.superlinked.com" and not (api_key and api_key.strip()):
        raise SystemExit("Set SIE_API_KEY or use --ask-key. Never put the key in source code.")
    payload = json.loads(args.samples.read_text())
    params = payload["watermark_params"]
    tok = AutoTokenizer.from_pretrained(payload["wm_model_id"])
    config = AutoConfig.from_pretrained(payload["wm_model_id"])
    detector = WatermarkDetector(
        model_config=config, device="cpu",
        watermarking_config=WatermarkingConfig(**params), ignore_repeated_ngrams=True,
    )
    proc = WatermarkLogitsProcessor(vocab_size=config.vocab_size, device="cpu", **params)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    client = SIEClient(args.sie_url, timeout_s=180.0,
                       api_key=api_key.strip() if api_key else None)

    def ids_of(t):
        return tok(t, add_special_tokens=False).input_ids

    def is_green(prev, cur):
        gl = proc._get_greenlist_ids(torch.tensor([[prev]]))
        flat = gl[0] if gl.dim() > 1 else gl
        return cur in set(flat.tolist())

    def detect_z(t):
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids
        return float(detector(ids, return_dict=True).z_score[0])

    available_arms = {
        "rt_ar (translation)": lambda c, t: round_trip(c, t, ["<2ar>"], "llm"),
        "paraphrase_1x": lambda c, t: paraphrase_llm(c, t),
        "paraphrase_2x": lambda c, t: paraphrase_recursive(c, t, 2),
        "paraphrase+rt_ar": paraphrase_then_translate,
    }
    arms = {name: fn for name, fn in available_arms.items()
            if ("rt_ar" if name == "rt_ar (translation)" else name) in args.arms}
    watermarked = [s for s in payload["samples"] if s["kind"] == "watermarked"][:args.limit]
    if not watermarked:
        client.close()
        raise SystemExit("No watermarked source passages in --samples.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump([], handle)

    base_zs = [detect_z(s["text"]) for s in watermarked]
    base_z = statistics.mean(base_zs)
    print(f"baseline meanz={base_z:5.2f} det={sum(z > Z for z in base_zs)}/{len(base_zs)}")
    print(f"{'arm':20s} {'meanz':>6s} {'det':>5s} {'chrf':>5s} {'cos':>5s} "
          f"{'ret%':>5s} {'g|ret':>6s} {'g|nov':>6s}")

    all_rows = []
    for name, fn in arms.items():
        zs, chrfs, coss = [], [], []
        ret_c, ret_g, nov_c, nov_g = 0, 0, 0, 0
        for s in watermarked:
            try:
                attacked = fn(client, s["text"])
            except Exception as exc:
                client.close()
                raise SystemExit(
                    f"{name}, {s['id']}: {exc}. Stopped; completed outputs are in {args.output}."
                ) from None
            zs.append(detect_z(attacked))
            chrfs.append(chrf.sentence_score(attacked, [s["text"]]).score)
            e = embedder.encode([s["text"], attacked], convert_to_tensor=True)
            coss.append(float(util.cos_sim(e[0], e[1])))
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
            all_rows.append({"arm": name, "id": s["id"], "z": zs[-1],
                             "chrf": chrfs[-1], "cos": coss[-1], "text": attacked,
                             "source_text": s["text"], "source_z": detect_z(s["text"]),
                             "watermark_params": params, "wm_model_id": payload["wm_model_id"],
                             "requested_transform_model": LLM_TRANSLATOR_MODEL})
            args.output.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n")
        if not zs:
            print(f"{name}: no successful outputs")
            continue
        total = ret_c + nov_c
        print(f"{name:20s} {statistics.mean(zs):6.2f} "
              f"{sum(1 for z in zs if z > Z):2d}/{len(zs):<2d} "
              f"{statistics.mean(chrfs):5.1f} {statistics.mean(coss):5.2f} "
              f"{100*ret_c/total:5.1f} {ret_g/max(ret_c,1):6.3f} {nov_g/max(nov_c,1):6.3f}")
    client.close()
    print(f"Saved {len(all_rows)} transformed records to {args.output}; archived results were unchanged.")


if __name__ == "__main__":
    main()
