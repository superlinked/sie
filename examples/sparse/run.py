#!/usr/bin/env python3
"""Send the /sparse-embeddings calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check      # offline, no key, the default
    python3 run.py --record     # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/encode/<model>
Models        prithivida/Splade_PP_en_v2   (Hugging Face f0d4aa214dcb60c274052a52c0497535e3aec64c)
              BAAI/bge-m3:sparse           (Hugging Face 5617a9f61b028005a4858fdac845db406aefb181)

Standard library only.

Thirteen texts go to both models, so 26 calls. Every call asks for
`output_types: ["sparse"]` and gets back token indices with weights.

`--check` makes no network call. It rebuilds all 26 recorded request bodies
from `data/inputs/inputs.json` and compares each with the recorded request.

`--record` writes calls.json only. It does NOT rewrite `derived/decoded/`,
which maps token indices back to strings using each model's own tokenizer. That
needs the tokenizer files, and this example is built to run without downloading
model weights.

Migrated from apps/site/tests/fixtures/reference/sparse/run.py in
superlinked/sie-web@b07b6d73.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://api.superlinked.com"
MODELS = {
    "splade": "prithivida/Splade_PP_en_v2",
    "bge": "BAAI/bge-m3:sparse",
}
# The folder each model's decoded terms live under in the dataset.
DECODED_DIR = {"splade": "splade", "bge": "bge-m3-sparse"}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def build_body(input_id: str, text: str) -> dict[str, Any]:
    return {"items": [{"id": input_id, "text": text}], "params": {"output_types": ["sparse"]}}


def check(data_dir: Path) -> int:
    texts = {item["id"]: item["text"] for item in load(data_dir / "inputs/inputs.json")["inputs"]}
    calls = load(data_dir / "calls.json")["calls"]

    rebuilt = 0
    mismatched: list[str] = []
    for call in calls:
        text = texts.get(call["case"])
        if text is None:
            mismatched.append(f"{call['id']}: no input with that id in inputs/inputs.json")
            continue
        if build_body(call["case"], text) != call["request"]["body"]:
            mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")
            continue
        if call["model"] not in MODELS.values():
            mismatched.append(f"{call['id']}: unexpected model {call['model']}")
            continue
        rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched else 0


def post(url: str, key: str, body: dict[str, Any]) -> tuple[int, dict[str, str], Any, float]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            raw, status, headers = response.read(), response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        raw, status, headers = error.read(), error.code, dict(error.headers)
    return status, headers, json.loads(raw), round((time.monotonic() - started) * 1000, 1)


def record(data_dir: Path, out_path: Path) -> int:
    key = os.environ.get("SIE_API_KEY", "").strip()
    if not key:
        raise SystemExit("--record needs SIE_API_KEY. Use --check for the offline check.")
    endpoint = (os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT).rstrip("/")
    inputs = load(data_dir / "inputs/inputs.json")["inputs"]

    calls = []
    for set_name, model in MODELS.items():
        path = f"/v1/encode/{model}"
        for item in inputs:
            body = build_body(item["id"], item["text"])
            status, headers, response, latency_ms = post(f"{endpoint}{path}", key, body)
            calls.append(
                {
                    "id": f"{DECODED_DIR[set_name]}/{item['id']}",
                    "set": set_name,
                    "case": item["id"],
                    "model": model,
                    "endpoint": endpoint,
                    "path": path,
                    "status": status,
                    "timing": {"latency_ms": latency_ms, "attempts": 1},
                    "request": {
                        "method": "POST",
                        "endpoint": endpoint,
                        "path": path,
                        "model": model,
                        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
                        "body": body,
                    },
                    "response": {"status": status, "body": response},
                    "recorded": {"model_revision": headers.get("x-sie-model-revision")},
                }
            )
            print(f"{model} {item['id']}: HTTP {status} in {latency_ms:.0f}ms")

    out_path.write_text(
        json.dumps({"task": "sparse", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print("derived/decoded/ is NOT rewritten by --record; it needs each model's tokenizer")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not args.record:
        return check(data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return record(data_dir, out_path)


if __name__ == "__main__":
    sys.exit(main())
