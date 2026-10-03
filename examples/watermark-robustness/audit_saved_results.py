"""Re-score saved texts offline; preserve the original experiment files.

Run with HF_HUB_OFFLINE=1. No translation or generation calls are made.
Output: review-results.json. Strength sweep texts were not saved and cannot
be re-scored; that limitation is recorded in the output.
"""

import json
from collections import defaultdict
from statistics import mean

from transformers import AutoConfig, AutoTokenizer, WatermarkingConfig

from wm_common import HERE, SAMPLES_PATH
from wm_detector import WatermarkDetector


def main():
    payload = json.loads(SAMPLES_PATH.read_text())
    tok = AutoTokenizer.from_pretrained(payload["wm_model_id"], local_files_only=True)
    config = AutoConfig.from_pretrained(payload["wm_model_id"], local_files_only=True)
    detector = WatermarkDetector(
        model_config=config, device="cpu", max_cache_size=10000,
        watermarking_config=WatermarkingConfig(**payload["watermark_params"]),
        ignore_repeated_ngrams=True,
    )
    originals = {s["id"]: s for s in payload["samples"]}

    def score(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        out = detector(ids, return_dict=True)
        pairs = set(zip(ids[0].tolist()[:-1], ids[0].tolist()[1:]))
        return {
            "z": float(out.z_score[0]), "p_normal": float(out.p_value[0]),
            "tokens_scored": int(out.num_tokens_scored[0]),
            "green_tokens": int(out.num_green_tokens[0]),
            "green_fraction": float(out.green_fraction[0]),
            "token_count": ids.shape[1],
        }, pairs

    baseline = {sid: score(s["text"]) for sid, s in originals.items()}
    records = []
    for filename in ["results.json", "results-cloud-qwen.json",
                     "paraphrase-results.json", "distance-sweep-results.json"]:
        for row in json.loads((HERE / filename).read_text()):
            sid = row["id"]
            scored, pairs = score(row["text"])
            base, original_pairs = baseline[sid]
            retained = pairs & original_pairs
            novel = pairs - original_pairs

            def green_count(bigrams):
                return sum(detector._get_ngram_score_cached((a,), b) for a, b in bigrams)

            retained_green, novel_green = green_count(retained), green_count(novel)
            assert len(retained) + len(novel) == scored["tokens_scored"]
            assert retained_green + novel_green == scored["green_tokens"]
            records.append({
                "file": filename, "id": sid, "kind": originals[sid]["kind"],
                "arm": row.get("arm", row.get("pivot", "unknown")),
                "legacy_z": row["z"], **scored,
                "length_ratio": scored["token_count"] / base["token_count"],
                "retained_fraction_of_output": len(retained) / len(pairs),
                "retained_fraction_of_original": len(retained) / len(original_pairs),
                "retained_count": len(retained), "retained_green": retained_green,
                "novel_count": len(novel), "novel_green": novel_green,
                "cosine": row.get("cosine", row.get("cos")),
                "chrf": row.get("chrf"),
            })
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["file"], row["kind"], row["arm"])].append(row)
    summary = []
    for (filename, kind, arm), rows in grouped.items():
        ret = sum(r["retained_count"] for r in rows)
        nov = sum(r["novel_count"] for r in rows)
        cosines = [r["cosine"] for r in rows if r["cosine"] is not None]
        item = {
            "file": filename, "kind": kind, "arm": arm, "n": len(rows),
            "mean_z": mean(r["z"] for r in rows),
            "detected": sum(r["z"] > 3 for r in rows),
            "mean_cosine": mean(cosines) if cosines else None,
            "mean_length_ratio": mean(r["length_ratio"] for r in rows),
            "min_length_ratio": min(r["length_ratio"] for r in rows),
            "retained_fraction_of_output_pooled": ret / (ret + nov),
            "green_rate_retained": sum(r["retained_green"] for r in rows) / ret if ret else None,
            "green_rate_novel": sum(r["novel_green"] for r in rows) / nov if nov else None,
        }
        summary.append(item)
        print(f"{filename:28} {kind:11} {arm:20} n={len(rows)} "
              f"z={item['mean_z']:.3f} detected={item['detected']}/{len(rows)} "
              f"retained={item['retained_fraction_of_output_pooled']:.3f}")
    output = {
        "method": "CPU green lists; unique integer-tuple n-grams; z > 3; nominal normal tails",
        "limitations": [
            "Controls are unwatermarked model outputs, not human-written text.",
            "Only eight source prompts; transformations of the same source are dependent.",
            "Legacy strength sweep stores scores only, so corrected rescoring is impossible.",
            "Cosine similarities are archived proxies, not factual equivalence judgments.",
        ],
        "baseline": {sid: result[0] for sid, result in baseline.items()},
        "summary": summary, "records": records,
    }
    (HERE / "review-results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
