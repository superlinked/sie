#!/usr/bin/env python3
"""Read business documents into their own JSON schema with SIE's Qwen3.8 27B.

    python3 fetch.py                        # the recorded run: schemas, expected values, every reply
    export SIE_API_KEY=sk-sie-...
    uv run run.py --docs 20                 # 20 documents spread over the set, a few cents
    uv run run.py --all                     # all 800 documents, about $2
    uv run run.py --show omni-100           # print one request without sending it

Every document goes to SIE Cloud's OpenAI-compatible chat route as one call: the system prompt, the page image, and
the document's own schema as a strict `json_schema` response format, greedy. These are the settings the recorded run
used for every model. Replies land in runs/sie-qwen3.8-27b.jsonl, which `score.py --runs runs` and
`node score.mjs runs` score the same way the recorded run was scored.

The page images come from their own datasets at the revisions the recorded run used: the OmniAI OCR Benchmark (MIT)
and CORD v2's test split (CC-BY-4.0), read through the Hugging Face dataset viewer as the recorded run read it. Each is checked against the sha256 the run pinned before any call.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
IMAGES = HERE / "images"
RUNS = HERE / "runs"
ARM = "sie-qwen3.8-27b"
MODEL = "Qwen/Qwen3.8-27B-FP8"
BASE_URL = "https://api.superlinked.com/v1"
MAX_TOKENS = 4096
OMNI_FILE = "https://huggingface.co/datasets/getomni-ai/ocr-benchmark/resolve/{rev}/test/images/{name}"
# CORD's test split through the Hugging Face dataset viewer, which is where the recorded run read it.
CORD_ROWS = "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix/cord-v2&config=default&split=test&offset={o}&length=1"


def strict_schema(s: dict[str, Any]) -> dict[str, Any]:
    """The one schema every model received: every key required, every value nullable, no extra keys.

    Strict structured outputs need every property required, so a field the document lacks is expressed as null, which
    is how the benchmark's expected values record it. Keywords strict modes reject (`not`, `pattern`, `format`) are
    dropped, and the benchmark's misspelt `emum` is read as `enum`. The expected values are not touched.
    """
    t = s.get("type")
    if t == "enum" or (t is None and "enum" in s):
        t = "string"
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    out: dict[str, Any] = {}
    if s.get("description"):
        out["description"] = s["description"]
    if t == "object":
        props = {k: strict_schema(v) for k, v in (s.get("properties") or {}).items()}
        return out | {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    if t == "array":
        items = s.get("items") if isinstance(s.get("items"), dict) else {"type": "string"}
        return {"anyOf": [out | {"type": "array", "items": strict_schema(items)}, {"type": "null"}]}
    leaf: dict[str, Any] = {"type": t or "string"}
    enum = s.get("enum") or s.get("emum")
    if enum and all(isinstance(e, str) for e in enum):
        leaf["enum"] = list(enum)
    return {"anyOf": [out | leaf, {"type": "null"}]}


def mime(image: bytes) -> str:
    if image.startswith(b"\x89PNG"):
        return "image/png"
    if image[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    raise ValueError("unknown image type")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sie-examples/doc-field-extraction-omni"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def image_for(doc: dict[str, Any], omni_revision: str) -> bytes:
    """The document's page image, from its source dataset, checked against the pinned sha256."""
    path = IMAGES / f"{doc['id']}{Path(doc['image']).suffix}"
    if not path.exists():
        IMAGES.mkdir(exist_ok=True)
        if doc["set"] == "omni":
            data = fetch(OMNI_FILE.format(rev=omni_revision, name=doc["source_file"]))
            if doc.get("reencoded"):
                # One page exceeds Anthropic's 5 MB limit; the run re-encoded it once, for every model alike.
                from PIL import Image

                buf = io.BytesIO()
                Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=90)
                data = buf.getvalue()
        else:
            data = cord_image(int(doc["id"].split("-")[1]))
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != doc["sha256"]:
        raise SystemExit(f"{doc['id']}: the image does not match the sha256 the recorded run pinned")
    return data


def cord_image(index: int) -> bytes:
    row = json.loads(fetch(CORD_ROWS.format(o=index)))["rows"][0]["row"]
    return fetch(row["image"]["src"])


def request_body(doc: dict[str, Any], image: bytes, system: str) -> dict[str, Any]:
    b64 = base64.b64encode(image).decode()
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime(image)};base64,{b64}"}}]},
        ],
        "max_completion_tokens": MAX_TOKENS,
        "temperature": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "fields", "schema": strict_schema(doc["schema"]), "strict": True}},
    }  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--docs", type=int, help="this many documents, spread evenly over the set")
    group.add_argument("--all", action="store_true")
    group.add_argument("--show", metavar="ID", help="print one request, image elided, without sending it")
    ap.add_argument("-c", "--concurrency", type=int, default=4)
    args = ap.parse_args()

    sets_path = EVIDENCE / "inputs" / "sets.json"
    if not sets_path.exists():
        raise SystemExit("run python3 fetch.py first")
    sets = json.loads(sets_path.read_text(encoding="utf-8"))
    docs = sets["documents"]
    if args.show:
        doc = next((d for d in docs if d["id"] == args.show), None)
        if doc is None:
            raise SystemExit(f"no document {args.show}")
        body = request_body(doc, image_for(doc, sets["omni_revision"]), sets["system"])
        body["messages"][1]["content"][0]["image_url"]["url"] = "data:…(the page image)…"
        print(json.dumps(body, indent=1))
        return 0
    if args.docs:
        step = max(1, len(docs) // args.docs)
        docs = docs[::step][: args.docs]

    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key=os.environ["SIE_API_KEY"], max_retries=3, timeout=900)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{ARM}.jsonl"

    def one(doc: dict[str, Any]) -> dict[str, Any]:
        body = request_body(doc, image_for(doc, sets["omni_revision"]), sets["system"])
        sent = time.perf_counter()
        try:
            resp = client.chat.completions.create(**body)
            row = {"text": resp.choices[0].message.content or "", "finish": resp.choices[0].finish_reason,
                   "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens, "error": None}  # fmt: skip
        except Exception as exc:  # noqa: BLE001 -- recorded and scored as no reply
            row = {"text": "", "error": repr(exc)[:300]}
        return {"id": doc["id"], "seconds": round(time.perf_counter() - sent, 3), **row}

    with ThreadPoolExecutor(args.concurrency) as pool, out.open("w", encoding="utf-8") as f:
        futures = [pool.submit(one, d) for d in docs]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{n}/{len(docs)} {row['id']} {row.get('finish') or row['error']}", flush=True)
    print(f"Wrote {out}. Score it: python3 score.py --runs runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
