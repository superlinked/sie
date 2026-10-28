"""Translate one product string into Japanese with tencent/Hy-MT2-1.8B on SIE.

    uv run --frozen python run.py --show   # print the request, send nothing
    uv run --frozen python run.py          # send it (needs SIE_API_KEY or SIE_BASE_URL)

One native generate call, the call https://superlinked.com/translation shows:

    POST /v1/generate/tencent__Hy-MT2-1.8B
    {"prompt": "<Hy-MT2 turn markers, instruction naming Japanese, source text>",
     "max_new_tokens": 256, "temperature": 0}

max_new_tokens grows with the input the way the page sizes it: 256 for the
recorded string, up to the model's 1024 for longer text. A translation that
stops at that cap (finish reason "length") is cut off, so the run exits 1.

The key comes from SIE_API_KEY and is never printed. Without SIE_BASE_URL the
SDK sends the call to SIE Serverless, https://api.superlinked.com.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
RECORDED = json.loads((HERE / "recorded.json").read_text(encoding="utf-8"))

MODEL = "tencent/Hy-MT2-1.8B"
# Native generate takes the SIE-safe id; the SDK normalizes it.
PATH = f"/v1/generate/{MODEL.replace('/', '__')}"
SERVERLESS = "https://api.superlinked.com"
# The output cap, as the page computes it: 5 output tokens per 3 input
# characters plus 16, rounded up to the next bucket, never above the model's
# declared 1024.
OUTPUT_TOKEN_BUCKETS = (64, 128, 256, 512, 1024)
OUTPUT_EXPANSION = 5
MAX_OUTPUT_TOKENS = 1024
# The model profile defaults to temperature 0.7, so the call asks for 0.
TEMPERATURE = 0
# The raw prompt for /v1/generate: Hy-MT2's chat turn markers around the
# English instruction from its model card. The recorded call puts one space
# between the instruction and the text, where the card puts a blank line.
PROMPT = (
    "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into {language}. "
    "Note that you should only output the translated result without any additional explanation: "
    "{text}<｜hy_Assistant｜>"
)


def build_prompt(text: str, language: str) -> str:
    """Hy-MT2's prompt for one source text, edge whitespace removed."""
    return PROMPT.format(language=language, text=text.strip())


def output_cap(text: str) -> int:
    """max_new_tokens for one source text; len() counts code points, as the page does."""
    raw = OUTPUT_EXPANSION * math.ceil(len(text.strip()) / 3) + 16
    bucket = next((size for size in OUTPUT_TOKEN_BUCKETS if size >= raw), OUTPUT_TOKEN_BUCKETS[-1])
    return min(bucket, MAX_OUTPUT_TOKENS)


def request_body(text: str, language: str) -> dict[str, Any]:
    return {"prompt": build_prompt(text, language), "max_new_tokens": output_cap(text), "temperature": TEMPERATURE}


def translate(client: SIEClient, text: str, language: str) -> dict[str, Any]:
    """Send one generate call and keep the fields a caller reads."""
    result = client.generate(
        MODEL,
        build_prompt(text, language),
        max_new_tokens=output_cap(text),
        temperature=TEMPERATURE,
        # Wait up to 300 s for capacity. Hy-MT2's profile allows a request up to
        # 600 s, and a per-call read timeout is not capped by the capacity wait.
        provision_timeout_s=300,
        read_timeout_s=600,
    )
    if not isinstance(result.get("text"), str):
        raise ValueError("The response has no text")
    usage = result.get("usage") or {}
    # The full response also carries request ids, timings and billing fields;
    # only the translation and its token counts are printed.
    return {
        "model": result.get("model"),
        "text": result["text"],
        "finish_reason": result.get("finish_reason"),
        "usage": {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens")},
    }


def is_recorded_case(text: str, language: str) -> bool:
    source = RECORDED["source"]
    return text.strip() == source["text"] and language == source["translate_into"]


def main(argv: list[str] | None = None) -> int:
    source = RECORDED["source"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="print the request and exit, sending nothing")
    parser.add_argument("--text", default=source["text"], help="English text to translate (default: the recorded one)")
    parser.add_argument(
        "--into", default=source["translate_into"], help="target language name, in English (default: Japanese)"
    )
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error("--text is empty")

    base_url = os.environ.get("SIE_BASE_URL", "").strip()
    if args.show:
        request = {"method": "POST", "url": (base_url or SERVERLESS).rstrip("/") + PATH}
        print(json.dumps({**request, "body": request_body(args.text, args.into)}, ensure_ascii=False, indent=2))
        return 0
    if not base_url and not os.environ.get("SIE_API_KEY", "").strip():
        parser.error("set SIE_API_KEY for SIE Serverless, or SIE_BASE_URL for your own server")

    try:
        with SIEClient() as client:
            output = translate(client, args.text, args.into)
    except Exception as exc:  # noqa: BLE001 - SDK error text can carry request context; print only its kind.
        status = getattr(exc, "status_code", None)
        detail = f"{type(exc).__name__}" + (f", HTTP {status}" if status else "")
        print(f"The translation call failed ({detail}).", file=sys.stderr)
        return 1

    if is_recorded_case(args.text, args.into):
        recorded = RECORDED["recorded"]
        output["matches_recorded"] = output["text"] == recorded["response"]["text"]
        output["recorded_on"] = recorded["date"]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["finish_reason"] == "length":
        print(
            f"The translation was cut off at the {output_cap(args.text)}-token output cap (finish reason length).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
