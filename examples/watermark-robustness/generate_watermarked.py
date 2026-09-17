"""Generate watermarked and control samples locally.

Produces samples.json: N watermarked generations (green-list watermark via
transformers' WatermarkingConfig) plus N unwatermarked controls from the
same prompts and sampling settings. All text is our own output from a model
we run with a key we hold; that is the point of the experiment.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig

from wm_common import PROMPTS, SAMPLES_PATH, WATERMARK_PARAMS, WM_MODEL_ID


def pick_device() -> str:
    # The watermark greenlist RNG is device-dependent: generation and
    # detection must run on the same device class, and the detector runs
    # on CPU. MPS generation produces green lists the CPU detector cannot
    # reproduce (measured: z ~= 0 on watermarked text).
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=SAMPLES_PATH.parent / "runs/generated/samples.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output; saved evidence is never overwritten.")
    if args.max_new_tokens < 1 or args.temperature <= 0:
        parser.error("Token limit and temperature must be positive.")

    device = pick_device()
    print(f"loading {WM_MODEL_ID} on {device}")
    tok = AutoTokenizer.from_pretrained(WM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        WM_MODEL_ID,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()

    wm_cfg = WatermarkingConfig(**WATERMARK_PARAMS)
    samples = []
    for i, user_prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": user_prompt}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        for kind, wm in [("watermarked", wm_cfg), ("control", None)]:
            torch.manual_seed(1000 + i)
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.1,
                watermarking_config=wm,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
            generated_ids = out[0, prompt_len:].tolist()
            samples.append({
                "id": f"{kind[:2]}-{i}", "kind": kind, "prompt": user_prompt, "text": text,
                "seed": 1000 + i, "prompt_token_ids": inputs.input_ids[0].tolist(),
                "generated_token_ids": generated_ids,
                "reached_token_cap": len(generated_ids) == args.max_new_tokens,
            })
            print(f"  [{kind:11s}] prompt {i}: {len(text.split())} words")

    payload = {
        "wm_model_id": WM_MODEL_ID,
        "watermark_params": WATERMARK_PARAMS,
        "generation_settings": {"temperature": args.temperature, "top_p": 0.95,
                                "top_k": 20, "repetition_penalty": 1.1,
                                "max_new_tokens": args.max_new_tokens, "do_sample": True},
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "device": device, "model_revision": getattr(model.config, "_commit_hash", None)},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
