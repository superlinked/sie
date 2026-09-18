"""Run the robustness eval: detect the watermark before and after each attack arm.

Arms (watermarked samples only; controls establish the detector's null):
  identity     no transformation (baseline detection power)
  rt_ar        en -> ar -> en round trip
  rt_ar_hu     en -> ar -> hu -> en (does a third hop add anything?)
  ner_rt_ar    entity-protected en -> ar -> en (GLiNER placeholders)

For every (sample, arm) we report the detector z-score and p-value, chrF++
against the original (surface overlap), and MiniLM embedding cosine similarity
(a proxy, not validated semantic equivalence). Writes results.json and report.md.
"""

import argparse
import json
import os
import statistics

import torch
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer, util
from sie_sdk import SIEClient
from transformers import AutoConfig, AutoTokenizer, WatermarkingConfig

from wm_detector import WatermarkDetector

from arms import LLM_TRANSLATOR_MODEL, ner_protected_round_trip, round_trip
from wm_common import REPORT_PATH, RESULTS_PATH, SAMPLES_PATH, SIE_URL

Z_THRESHOLD = 3.0
chrf = CHRF(word_order=2)


def detect(detector, tok, text: str) -> dict:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    out = detector(ids, return_dict=True)
    return {
        "z": float(out.z_score[0]),
        "p": float(out.p_value[0]),
        "green_fraction": float(out.green_fraction[0]),
        "tokens_scored": int(out.num_tokens_scored[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sie-url", default=SIE_URL)
    parser.add_argument("--arms", default="identity,rt_ar,rt_ar_hu,ner_rt_ar")
    parser.add_argument("--translator", default="madlad", choices=["madlad", "llm"])
    parser.add_argument("--suffix", default="", help="appended to results/report filenames")
    args = parser.parse_args()
    arm_names = args.arms.split(",")
    results_path = RESULTS_PATH.with_stem(RESULTS_PATH.stem + args.suffix)
    report_path = REPORT_PATH.with_stem(REPORT_PATH.stem + args.suffix)

    payload = json.loads(SAMPLES_PATH.read_text())
    wm_cfg = WatermarkingConfig(**payload["watermark_params"])
    tok = AutoTokenizer.from_pretrained(payload["wm_model_id"])
    model_config = AutoConfig.from_pretrained(payload["wm_model_id"])
    detector = WatermarkDetector(
        model_config=model_config,
        device="cpu",
        watermarking_config=wm_cfg,
        ignore_repeated_ngrams=True,
    )
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    client = SIEClient(args.sie_url, timeout_s=600.0, api_key=os.environ.get("SIE_API_KEY"))

    def transform(arm: str, text: str) -> tuple[str, dict]:
        if arm == "identity":
            return text, {}
        if arm == "rt_ar":
            return round_trip(client, text, ["<2ar>"], args.translator), {}
        if arm == "rt_ar_hu":
            return round_trip(client, text, ["<2ar>", "<2hu>"], args.translator), {}
        if arm == "ner_rt_ar":
            return ner_protected_round_trip(client, text, ["<2ar>"], args.translator)
        raise ValueError(arm)

    rows = []
    failures = 0
    for sample in payload["samples"]:
        sample_arms = arm_names if sample["kind"] == "watermarked" else ["identity"]
        for arm in sample_arms:
            try:
                attacked, extra = transform(arm, sample["text"])
            except Exception as exc:
                failures += 1
                rows.append({"id": sample["id"], "kind": sample["kind"],
                             "arm": arm, "error": str(exc)})
                print(f"{sample['id']:6s} {arm:10s} FAILED: {exc}")
                continue
            row = {"id": sample["id"], "kind": sample["kind"], "arm": arm, **extra}
            row.update(detect(detector, tok, attacked))
            if arm != "identity":
                row["chrf"] = chrf.sentence_score(attacked, [sample["text"]]).score
                embs = embedder.encode([sample["text"], attacked], convert_to_tensor=True)
                row["cosine"] = float(util.cos_sim(embs[0], embs[1]))
            row["text"] = attacked
            rows.append(row)
            print(f"{sample['id']:6s} {arm:10s} z={row['z']:6.2f} p={row['p']:.2e}"
                  + (f" chrf={row['chrf']:.1f} cos={row['cosine']:.2f}" if arm != "identity" else ""))

    try:
        with results_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(rows, indent=2, ensure_ascii=False))
    except FileExistsError:
        raise SystemExit(
            f"{results_path} already exists; pass a new --suffix so recorded "
            "evidence is never replaced."
        )
    if failures:
        raise SystemExit(
            f"{failures} sample-arm(s) failed; wrote {results_path} with explicit "
            "failure records and skipped the report. This partial run is not a "
            "replacement for the published results."
        )
    write_report(rows, payload, arm_names, report_path, args.translator)
    print(f"wrote {results_path} and {report_path}")


def agg(rows, key):
    vals = [r[key] for r in rows if key in r]
    return statistics.mean(vals) if vals else float("nan")


def write_report(rows, payload, arm_names, report_path, translator) -> None:
    wm = [r for r in rows if r["kind"] == "watermarked"]
    ctl = [r for r in rows if r["kind"] == "control"]
    translator_desc = (
        "SIE `google/madlad400-3b-mt` (CTranslate2, greedy, sentence-level)"
        if translator == "madlad"
        else f"SIE `{LLM_TRANSLATOR_MODEL}` (instruction-prompted LLM translation, greedy, whole-text)"
    )
    lines = [
        "# Watermark robustness under round-trip translation",
        "",
        f"Watermark: transformers green-list (`{json.dumps(payload['watermark_params'])}`) "
        f"on `{payload['wm_model_id']}`. Translation: {translator_desc}. "
        "NER: SIE `urchade/gliner_multi-v2.1`. "
        f"Detection threshold z > {Z_THRESHOLD}.",
        "",
        "| arm | n | mean z | mean p | detected | mean chrF++ | mean cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in arm_names:
        sub = [r for r in wm if r["arm"] == arm]
        if not sub:
            continue
        detected = sum(1 for r in sub if r["z"] > Z_THRESHOLD)
        chrf_s = f"{agg(sub, 'chrf'):.1f}" if arm != "identity" else "-"
        cos_s = f"{agg(sub, 'cosine'):.2f}" if arm != "identity" else "-"
        lines.append(
            f"| {arm} | {len(sub)} | {agg(sub, 'z'):.2f} | {agg(sub, 'p'):.2e} "
            f"| {detected}/{len(sub)} | {chrf_s} | {cos_s} |"
        )
    detected_ctl = sum(1 for r in ctl if r["z"] > Z_THRESHOLD)
    lines += [
        f"| control (unwatermarked) | {len(ctl)} | {agg(ctl, 'z'):.2f} | {agg(ctl, 'p'):.2e} "
        f"| {detected_ctl}/{len(ctl)} | - | - |",
        "",
    ]
    ner_rows = [r for r in wm if r["arm"] == "ner_rt_ar" and "entities" in r]
    if ner_rows:
        total_e = sum(r["entities"] for r in ner_rows)
        total_s = sum(r["placeholders_survived"] for r in ner_rows)
        lines += [f"NER arm: {total_s}/{total_e} entity placeholders survived the round trip.", ""]
    example = next((r for r in wm if r["arm"] == "rt_ar"), None)
    if example:
        original = next(s for s in payload["samples"] if s["id"] == example["id"])
        lines += [
            "## Example (rt_ar)",
            "",
            "**Original (synthetic test text; factual claims are not validated):** "
            f"{original['text']}",
            "",
            f"**Round-tripped (z={example['z']:.2f}):** {example['text']}",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- All input text is generated locally with a watermark key we hold; "
        "this measures the decay of our own watermark, the same methodology as "
        "the robustness sections of the watermarking papers themselves.",
        "- Small n; z-scores per sample matter more than the detected fraction.",
        "- chrF++ measures surface overlap; embedding cosine is a similarity proxy, "
        "not the fraction of meaning preserved. Check facts, omissions and truncation manually.",
        "- Controls are unwatermarked model outputs, not human-written text. "
        "Normal-tail p-values are nominal; eight controls cannot calibrate rare false alarms.",
    ]
    report_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
