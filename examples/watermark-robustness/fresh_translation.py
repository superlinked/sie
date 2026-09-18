"""Fresh, resumable local pilot: 21 prompts x 4 biases x 2 text versions.

Generation: cached Qwen2.5-1.5B, CPU, seven prompts per batch, three keys.
Transformation: cached MADLAD CT2 float32, English -> Arabic -> English.
The 168 scored records share 84 source generations and 21 prompt/key cells.
These are not 168 independent source prompts. All intermediate text is saved.
"""

import argparse
import csv
import json
import math
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig

from wm_common import HERE, WATERMARK_PARAMS, WM_MODEL_ID
from wm_detector import WatermarkDetector

PUBLISHED = HERE / "fresh-translation-2026-09-13"
OUT = PUBLISHED   # write directory; __main__ forbids writing into PUBLISHED
DATA = PUBLISHED  # read directory; equals OUT except in the report phase
LOCAL_ONLY = True
TRANSLATION_REPO = "superlinked/madlad400-3b-mt-ctranslate2-float32"
TRANSLATION_REVISION = "2cac811471ffb01ea6502ab4d317d5a7e9cb6f7d"
KEYS = [15485863, 32452843, 49979687]
BIASES = [0.0, 2.0, 3.0, 4.0]
TOPICS = [
    "explain how a neighborhood library can help people learn practical skills",
    "describe the atmosphere of a fictional seaside town on a rainy morning",
    "explain why urban trees provide shade and cool their surroundings",
    "argue for extending bus services into the evening, including one tradeoff",
    "describe how to plan a shared vegetable garden with neighbors",
    "write a fictional scene in which two friends repair an old bicycle",
    "explain why regular backups help protect important computer files",
    "explain how bread dough changes while yeast ferments it",
    "describe a fictional cafe that exchanges books as well as selling coffee",
    "explain why clear meeting notes can improve teamwork",
    "describe the experience of taking an overnight train, without naming real routes",
    "explain the difference between weather and climate for a beginner",
    "argue for teaching basic cooking in school, including one practical difficulty",
    "write a fictional review of a novel about a mapmaker who gets lost",
    "explain how a reusable shopping bag can reduce waste, including a limitation",
    "describe a fictional community workshop where people repair household objects",
    "explain why a fair comparison should change one experimental factor at a time",
    "describe the sounds and activity of a small market as it opens",
    "explain why translating a joke can be harder than translating a simple instruction",
    "argue for protecting a quiet space in a busy city, including one tradeoff",
    "explain how to distinguish an observation from an interpretation in a science notebook",
]
PROMPTS = [f"Write one complete paragraph of 70 to 90 words. {topic.capitalize()}. "
           "Use plain English. Do not include a title, list, or introduction about the task."
           for topic in TOPICS]


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(name, default):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else default


def params_for(bias, key):
    return {**WATERMARK_PARAMS, "bias": bias, "hashing_key": key}


def detector_for(config, bias, key):
    return WatermarkDetector(config, "cpu", WatermarkingConfig(**params_for(bias, key)),
                             ignore_repeated_ngrams=True, max_cache_size=4096)


def score(text, tokenizer, detector):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] < 2:
        return {"z": None, "tokens": ids.shape[1], "scored_pairs": 0, "green_hits": 0}
    result = detector(ids, return_dict=True)
    return {"z": float(result.z_score[0]), "tokens": ids.shape[1],
            "scored_pairs": int(result.num_tokens_scored[0]),
            "green_hits": int(result.num_green_tokens[0])}


def generate(limit_batches=None):
    tok = AutoTokenizer.from_pretrained(WM_MODEL_ID, local_files_only=LOCAL_ONLY, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(WM_MODEL_ID, local_files_only=LOCAL_ONLY, dtype=torch.float32).eval()
    rows = read("originals.json", [])
    completed = {r["id"] for r in rows}
    metadata = {
        "created_utc": read("manifest.json", {}).get("created_utc", datetime.now(timezone.utc).isoformat()),
        "design": "21 distinct prompts, 3 keys (one per seven prompts), 4 bias conditions, original plus Arabic round trip",
        "planned_scored_records": 168, "source_generations": 84,
        "model": WM_MODEL_ID, "model_revision": getattr(model.config, "_commit_hash", None),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "device": "cpu", "dtype": "float32",
        "threads": torch.get_num_threads(), "keys": KEYS, "biases": BIASES,
        # Immutable base watermark settings. params_for() overrides bias and
        # hashing_key per run; each source row records those effective values.
        "watermark_params": {k: v for k, v in WATERMARK_PARAMS.items()
                             if k not in {"bias", "hashing_key"}},
        "generation_settings": {"max_new_tokens": 224, "do_sample": True, "temperature": 0.8,
                                "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.1},
        "model_generation_defaults": model.generation_config.to_dict(),
        "prompts": PROMPTS,
        "translation_artifact": TRANSLATION_REPO, "translation_revision": TRANSLATION_REVISION,
    }
    existing_manifest = read("manifest.json", None)
    if existing_manifest is None:
        save("manifest.json", metadata)
    else:
        immutable = ["design", "planned_scored_records", "source_generations",
                     "model", "keys", "biases", "generation_settings", "prompts",
                     "watermark_params", "translation_artifact", "translation_revision"]
        changed = [k for k in immutable if existing_manifest.get(k) != metadata[k]]
        if changed:
            raise SystemExit(
                "manifest.json in the output directory records different settings "
                f"({', '.join(changed)}). Resume with the original configuration or "
                "choose a new --out-dir; the original provenance is not replaced."
            )
    batches_done = 0
    for group, key in enumerate(KEYS):
        indices = list(range(group * 7, group * 7 + 7))
        formatted = [tok.apply_chat_template([{"role": "user", "content": PROMPTS[i]}],
                     tokenize=False, add_generation_prompt=True) for i in indices]
        inputs = tok(formatted, padding=True, return_tensors="pt")
        plen = inputs.input_ids.shape[1]
        for bias in BIASES:
            ids = [f"prompt-{i:02d}-bias-{bias:g}" for i in indices]
            if all(sid in completed for sid in ids):
                continue
            if limit_batches is not None and batches_done >= limit_batches:
                return
            start = time.monotonic()
            seed = 20260913 + group
            torch.manual_seed(seed)
            wm = WatermarkingConfig(**params_for(bias, key)) if bias else None
            with torch.inference_mode():
                generated = model.generate(**inputs, **metadata["generation_settings"],
                                           watermarking_config=wm, pad_token_id=tok.pad_token_id)
            detector = detector_for(model.config, bias, key)
            eos_ids = model.generation_config.eos_token_id
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            for batch_index, (i, sid) in enumerate(zip(indices, ids)):
                completion = generated[batch_index, plen:].tolist()
                stop = next((j + 1 for j, token in enumerate(completion) if token in eos_ids), len(completion))
                completion = completion[:stop]
                text = tok.decode(completion, skip_special_tokens=True).strip()
                row = {
                    "id": sid, "prompt_index": i, "prompt": PROMPTS[i], "bias": bias, "key": key,
                    "batch_seed": seed, "batch_index": batch_index, "batch_prompt_indices": indices,
                    "prompt_token_ids": inputs.input_ids[batch_index].tolist(),
                    "attention_mask": inputs.attention_mask[batch_index].tolist(),
                    "generated_token_ids": completion, "text": text,
                    "finish_reason": "eos" if completion and completion[-1] in eos_ids else "length",
                    "detection": score(text, tok, detector),
                }
                if sid not in completed:
                    rows.append(row)
                    completed.add(sid)
            save("originals.json", rows)
            batches_done += 1
            print(f"generation {len(rows)}/84: bias={bias:g} group={group}, "
                  f"{time.monotonic() - start:.1f}s, mean tokens={statistics.mean(len(r['generated_token_ids']) for r in rows[-7:]):.0f}", flush=True)


def translate(limit_records=None):
    import ctranslate2
    from huggingface_hub import snapshot_download

    if ctranslate2.__version__ != "4.8.1":
        raise RuntimeError("This cached artifact requires CTranslate2 4.8.1.")
    if not read("originals.json", []):
        raise ValueError("No originals in --out-dir. Run the generate phase there first.")
    path = snapshot_download(TRANSLATION_REPO, revision=TRANSLATION_REVISION, local_files_only=LOCAL_ONLY)
    # This artifact contains a T5 Unigram/Metaspace tokenizer. Transformers
    # 4.57.6 misidentifies its CT2 config as Mistral when model_type is absent.
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    assert json.loads(tok.backend_tokenizer.to_str())["pre_tokenizer"]["type"] == "Metaspace"
    translator = ctranslate2.Translator(path, device="cpu", compute_type="float32",
                                       inter_threads=1, intra_threads=6)
    ctranslate2.set_random_seed(0)
    rows = read("translations.json", [])
    completed = {r["id"] for r in rows}
    settings = {"ctranslate2": ctranslate2.__version__, "beam_size": 1,
                "compute_type": "float32", "max_decoding_length": 256,
                "sampling_topk": 1, "sampling_topp": 1.0, "sampling_temperature": 1.0,
                "max_input_length": 0, "seed": 0, "intra_threads": 6}

    def step(texts, language):
        split = [[s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] for text in texts]
        segments = [segment for text_segments in split for segment in text_segments]
        encoded = [tok.convert_ids_to_tokens(tok.encode(f"<2{language}> {s}", add_special_tokens=True))
                   for s in segments]
        results = translator.translate_batch(encoded, beam_size=1, max_decoding_length=256,
                                             max_input_length=0, sampling_topk=1, sampling_topp=1.0,
                                             sampling_temperature=1.0, return_end_token=True)
        output = []
        for source, result in zip(segments, results, strict=True):
            tokens = result.hypotheses[0]
            output.append({"source": source, "output_tokens": tokens,
                           "text": tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True).strip(),
                           "finish_reason": "eos" if tokens and tokens[-1] == tok.eos_token else "length_or_other"})
        regrouped = []
        offset = 0
        for source_segments in split:
            rows_for_text = output[offset:offset + len(source_segments)]
            regrouped.append((" ".join(row["text"] for row in rows_for_text), rows_for_text))
            offset += len(source_segments)
        return regrouped

    pending = [r for r in read("originals.json", []) if r["id"] not in completed]
    if limit_records is not None:
        pending = pending[:limit_records]
    for offset in range(0, len(pending), 7):
        batch = pending[offset:offset + 7]
        start = time.monotonic()
        forward_batch = step([r["text"] for r in batch], "ar")
        backward_batch = step([r[0] for r in forward_batch], "en")
        for original, (arabic, forward), (english, backward) in zip(batch, forward_batch, backward_batch, strict=True):
            rows.append({"id": original["id"], "arabic": arabic, "text": english,
                         "forward_segments": forward, "backward_segments": backward,
                         "settings": settings, "batch_ids": [r["id"] for r in batch],
                         "batch_elapsed_seconds": time.monotonic() - start})
        save("translations.json", rows)
        print(f"translation {len(rows)}/84: {len(batch)} passages, {time.monotonic() - start:.1f}s", flush=True)


def report():
    from sentence_transformers import SentenceTransformer

    if not read("translations.json", []):
        raise ValueError("No translations in --out-dir. Run the translate phase there first.")
    tok = AutoTokenizer.from_pretrained(WM_MODEL_ID, local_files_only=LOCAL_ONLY)
    config = AutoConfig.from_pretrained(WM_MODEL_ID, local_files_only=LOCAL_ONLY)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=LOCAL_ONLY)
    originals = read("originals.json", [])
    translated = {r["id"]: r for r in read("translations.json", [])}
    if len(originals) != 84 or len(translated) != 84:
        raise RuntimeError("Complete all 84 originals and 84 round trips before producing the report.")
    if len({r["id"] for r in originals}) != 84:
        raise RuntimeError("Duplicate source records.")
    records, pairs = [], []
    for original in originals:
        detector = detector_for(config, original["bias"], original["key"])
        before = score(original["text"], tok, detector)
        if not math.isclose(before["z"], original["detection"]["z"], abs_tol=1e-10):
            raise RuntimeError("The generation and reporting environments disagree on the watermark score.")
        common = {"id": original["id"], "prompt_index": original["prompt_index"],
                  "key": original["key"], "bias": original["bias"]}
        records.append({**common, "arm": "original", "text": original["text"], **before})
        if original["id"] not in translated:
            continue
        transformed = translated[original["id"]]
        after = score(transformed["text"], tok, detector)
        records.append({**common, "arm": "arabic_round_trip", "text": transformed["text"], **after})
        embeddings = embedder.encode([original["text"], transformed["text"]], normalize_embeddings=True)
        cosine = float(embeddings[0] @ embeddings[1])
        embedding_ids = embedder.tokenizer([original["text"], transformed["text"]], truncation=False, verbose=False)["input_ids"]
        embedding_truncation = any(len(ids) > embedder.max_seq_length for ids in embedding_ids)
        length_ratio = after["tokens"] / before["tokens"] if before["tokens"] else None
        pairs.append({**common, "before": before, "after": after, "cosine": cosine,
                      "length_ratio": length_ratio, "source_finish_reason": original["finish_reason"],
                      "embedding_truncation_flag": embedding_truncation,
                      "translation_cap_flag": any(s["finish_reason"] != "eos" for s in transformed["forward_segments"] + transformed["backward_segments"]),
                      "length_review_flag": length_ratio is None or length_ratio < 0.7 or length_ratio > 1.4})
    summary = []
    for bias in BIASES:
        group = [r for r in pairs if r["bias"] == bias]
        if not group:
            continue
        valid = [r for r in group if r["before"]["z"] is not None and r["after"]["z"] is not None]
        summary.append({"bias": bias, "paired_n": len(group), "scorable_n": len(valid),
                        "mean_z_before": statistics.mean(r["before"]["z"] for r in valid),
                        "mean_z_after": statistics.mean(r["after"]["z"] for r in valid),
                        "detected_before": sum(r["before"]["z"] > 3 for r in valid),
                        "detected_after": sum(r["after"]["z"] > 3 for r in valid),
                        "mean_cosine": statistics.mean(r["cosine"] for r in group),
                        "mean_length_ratio": (statistics.mean(ratios)
                                              if (ratios := [r["length_ratio"] for r in group
                                                             if r["length_ratio"] is not None])
                                              else None),
                        "length_flags": sum(r["length_review_flag"] for r in group),
                        "source_cap_flags": sum(r["source_finish_reason"] != "eos" for r in group),
                        "embedding_truncation_flags": sum(r["embedding_truncation_flag"] for r in group),
                        "translation_cap_flags": sum(r["translation_cap_flag"] for r in group)})
        eligible = [r for r in valid if r["source_finish_reason"] == "eos"
                    and not r["translation_cap_flag"] and not r["length_review_flag"]]
        summary[-1]["passes_completeness_and_length_checks"] = {
            "n": len(eligible),
            "mean_z_before": statistics.mean(r["before"]["z"] for r in eligible) if eligible else None,
            "mean_z_after": statistics.mean(r["after"]["z"] for r in eligible) if eligible else None,
            "detected_before": sum(r["before"]["z"] > 3 for r in eligible),
            "detected_after": sum(r["after"]["z"] > 3 for r in eligible),
        }
    save("results.json", {"record_count": len(records), "summary": summary, "pairs": pairs, "records": records})
    with (OUT / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Fresh translation and watermark-strength pilot", "",
             "21 distinct prompts × four conditions × two versions = 168 scored records. "
             "The four conditions share prompts; these are 84 source generations, not 168 independent prompts.", "",
             "Generation: Qwen2.5-1.5B-Instruct on CPU; three experimental keys. "
             "Translation: local MADLAD-400-3B float32, English → Arabic → English. "
             "Detector counts unique token pairs; threshold is strictly z > 3.", "",
             "| Bias | Pairs | Mean z before | Mean z after | Detected before | Detected after | Mean length ratio | Mean cosine |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        n = row["paired_n"]
        label = "0 (unwatermarked)" if not row["bias"] else f"{row['bias']:g}"
        ratio_text = "n/a" if row["mean_length_ratio"] is None else f"{row['mean_length_ratio']:.2f}"
        lines.append(f"| {label} | {n} | {row['mean_z_before']:.2f} | {row['mean_z_after']:.2f} | "
                     f"{row['detected_before']}/{n} | {row['detected_after']}/{n} | "
                     f"{ratio_text} | {row['mean_cosine']:.3f} |")
    lines += ["", "All pairs are included; no results are removed for producing an inconvenient score.", "",
              f"Source token-cap flags: {sum(r['source_cap_flags'] for r in summary)}/84. "
              f"Translation stop flags: {sum(r['translation_cap_flags'] for r in summary)}/84. "
              f"Length-review flags (ratio below 0.7 or above 1.4): {sum(r['length_flags'] for r in summary)}/84.", "",
              f"Embedding input truncation flags: {sum(r['embedding_truncation_flags'] for r in summary)}/84. "
              "These pairs exceed the embedding model's input limit on at least one side.", "",
              "Cosine similarity is an embedding proxy, not factual equivalence or a percentage of meaning preserved. "
              "Length flags are prompts for review, not a semantic-quality test. "
              "Twenty-one controls per condition cannot calibrate a rare false-positive rate. "
              "The keys are assigned to groups of seven prompts, so key and prompt effects are not independently isolated.", "",
              "Full prompts/settings and library versions are in manifest.json. Exact source token IDs and finish reasons "
              "are in originals.json. Both translation directions, sentence outputs and batch membership are in translations.json. "
              "Per-pair scores and quality proxies are in results.json.", ""]
    lines += ["## Sensitivity check", "",
              "The primary table above includes every pair. The table below excludes source/translation stop flags "
              "and the predeclared length-review flags. Passing these checks does not establish semantic equivalence.", "",
              "| Bias | Pairs passing checks | Mean z before | Mean z after | Detected before | Detected after |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        check = row["passes_completeness_and_length_checks"]
        n = check["n"]
        if n:
            lines.append(f"| {row['bias']:g} | {n} | {check['mean_z_before']:.2f} | {check['mean_z_after']:.2f} | "
                         f"{check['detected_before']}/{n} | {check['detected_after']}/{n} |")
    lines += ["", "See manual-review.md for qualitative observations about this run. "
              "A low score on a failed translation is not evidence of meaning-preserving watermark removal.", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["generate", "translate", "report"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=HERE / "runs/fresh-translation")
    parser.add_argument("--dataset-dir", type=Path, default=PUBLISHED,
                        help="report phase: dataset to read (default: the published dataset).")
    parser.add_argument("--allow-downloads", action="store_true",
                        help="Allow missing model/tokenizer assets to download from Hugging Face.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.out_dir.resolve() == PUBLISHED.resolve():
        parser.error("The published dataset is read-only for every phase. Choose a new --out-dir.")
    OUT = args.out_dir
    DATA = args.dataset_dir if args.phase == "report" else args.out_dir
    LOCAL_ONLY = not args.allow_downloads
    torch.set_num_threads(6)
    if args.phase == "generate":
        generate(args.limit)
    elif args.phase == "translate":
        translate(args.limit)
    else:
        report()
