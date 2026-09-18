"""Prompt-conditioned, post-hoc KL analysis of the saved generations.

Replays the known Transformers 4.57.6 processing order: repetition penalty,
temperature, top-k, top-p, then watermark bias. Legacy samples lack the exact
generated IDs and settings, so their results remain reconstructed estimates.
Summed conditional KL along a realized path is not the sequence-level KL:
the chain rule also requires an expectation over watermarked prefixes.

Matching source bigrams only produces a KL-weighted overlap proxy. It does
not measure post-transformation distributional KL or certify robustness.
Writes bgen-review-results.json; preserves the original bgen-results.json.
"""

import json
import math
import statistics

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig
from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper, WatermarkLogitsProcessor,
)

from wm_common import HERE, SAMPLES_PATH
from wm_detector import WatermarkDetector


def green_tilt_kl(q, delta):
    """Exact KL of a green-list exponential tilt, for an already processed P."""
    if not 0 <= q <= 1:
        raise ValueError("green probability mass must be in [0, 1]")
    if q in (0, 1):
        return 0.0
    log_z = math.log1p(q * math.expm1(delta))
    q_watermarked = q * math.exp(delta - log_z)
    return max(0.0, delta * q_watermarked - log_z)


def main():
    payload = json.loads(SAMPLES_PATH.read_text())
    mid, params = payload["wm_model_id"], payload["watermark_params"]
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, local_files_only=True).eval()
    config = AutoConfig.from_pretrained(mid, local_files_only=True)
    wm = WatermarkLogitsProcessor(vocab_size=config.vocab_size, device="cpu", **params)
    detector = WatermarkDetector(config, "cpu", WatermarkingConfig(**params), ignore_repeated_ngrams=True)
    # Reconstructed from the original script and cached model configuration.
    settings = payload.get("generation_settings", {
        "temperature": 0.8, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.1,
    })
    processors = LogitsProcessorList()
    if settings["repetition_penalty"] != 1:
        processors.append(RepetitionPenaltyLogitsProcessor(settings["repetition_penalty"]))
    if settings["temperature"] != 1:
        processors.append(TemperatureLogitsWarper(settings["temperature"]))
    if settings["top_k"] > 0:
        processors.append(TopKLogitsWarper(settings["top_k"]))
    if settings["top_p"] < 1:
        processors.append(TopPLogitsWarper(settings["top_p"]))

    originals = {}
    for sample in payload["samples"]:
        if sample["kind"] != "watermarked":
            continue
        exact_ids = "prompt_token_ids" in sample and "generated_token_ids" in sample
        if exact_ids:
            prompt_ids, completion_ids = sample["prompt_token_ids"], sample["generated_token_ids"]
        else:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": sample["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tok(prompt).input_ids
            completion_ids = tok(sample["text"], add_special_tokens=False).input_ids
        ids = torch.tensor([prompt_ids + completion_ids])
        with torch.inference_mode():
            logits = model(ids).logits[0].float()
        contributions = []
        for i in range(len(prompt_ids), ids.shape[1]):
            prefix = ids[:, :i]
            scores = processors(prefix, logits[i - 1:i].clone())[0]
            probs = torch.softmax(scores.double(), dim=-1)
            green = wm._get_greenlist_ids(prefix[0])
            q = min(1.0, max(0.0, float(probs[green].sum())))
            kl = green_tilt_kl(q, float(params["bias"]))
            positive = probs[probs > 0]
            entropy = float(-(positive * positive.log()).sum())
            contributions.append({"bigram": ids[0, i - 1:i + 1].tolist(), "kl": kl, "entropy": entropy})
        text_ids = tok(sample["text"], return_tensors="pt", add_special_tokens=False).input_ids
        total = sum(r["kl"] for r in contributions)
        originals[sample["id"]] = {
            "sum_conditional_kl_nats": total,
            "input_provenance": "saved token IDs" if exact_ids else "reconstructed prompt and retokenized completion",
            "z": float(detector(text_ids, return_dict=True).z_score[0]),
            "rows": contributions,
        }
        print(f"{sample['id']}: conditional KL sum={total:.3f} nats ({originals[sample['id']]['input_provenance']})", flush=True)

    overlap = []
    for filename in ["results.json", "results-cloud-qwen.json", "paraphrase-results.json", "distance-sweep-results.json"]:
        for record in json.loads((HERE / filename).read_text()):
            if record["id"] not in originals or record.get("arm") == "identity":
                continue
            ids = tok(record["text"], add_special_tokens=False).input_ids
            pairs = set(zip(ids[:-1], ids[1:]))
            source = originals[record["id"]]
            retained = sum(r["kl"] for r in source["rows"] if tuple(r["bigram"]) in pairs)
            scored = detector(torch.tensor([ids]), return_dict=True)
            overlap.append({
                "file": filename, "id": record["id"],
                "arm": record.get("arm", record.get("pivot", "unknown")),
                "kl_weighted_bigram_overlap": retained / source["sum_conditional_kl_nats"] if source["sum_conditional_kl_nats"] else None,
                "z": float(scored.z_score[0]),
            })
    valid = [r for r in overlap if r["kl_weighted_bigram_overlap"] is not None]
    correlation = statistics.correlation([r["kl_weighted_bigram_overlap"] for r in valid], [r["z"] for r in valid]) if len(valid) > 1 else None
    output = {
        "status": "post-hoc reconstruction; not a generation-time measurement or robustness certificate",
        "settings": settings, "processing_order": "repetition penalty -> temperature -> top-k -> top-p -> watermark",
        "limitations": [
            "Legacy settings, model revision and exact generated token IDs were not recorded.",
            "Teacher-forced full-sequence evaluation can differ numerically from cached incremental generation.",
            "KL-weighted overlap is not retained distributional information.",
            "Pooled correlation reuses eight source texts and mixes translators and transformations.",
        ],
        "pooled_descriptive_overlap_z_correlation": correlation,
        "originals": originals, "overlap_rows": overlap,
    }
    (HERE / "bgen-review-results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
