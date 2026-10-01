#!/usr/bin/env python3
"""Inspect or send one recorded named-object detection request with SIE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
MODEL = "google/owlv2-base-patch16-ensemble"
REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
MANIFEST_SHA256 = "4ea377268746f0b19cb29ab1407f4ccdeb120c77c9218db6a7a0ed7528a312b4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", metavar="CASE", help="Print a request descriptor without sending a request")
    parser.add_argument("--case", default="countingpills-12", help="Send one selected image, default: countingpills-12")
    args = parser.parse_args()
    manifest_body = (EVIDENCE / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_body).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Evidence manifest differs from the immutable example pin")
    manifest = json.loads(manifest_body)
    page_body = (EVIDENCE / "page.json").read_bytes()
    page_pin = manifest["files"]["page.json"]
    if len(page_body) != page_pin["bytes"] or hashlib.sha256(page_body).hexdigest() != page_pin["sha256"]:
        raise ValueError("Request descriptor differs from the immutable evidence")
    page = json.loads(page_body)
    selected = args.show or args.case
    case = next((r for r in page["shown"] if f"{r['dataset']}-{r['imageId']}" == selected), None)
    if case is None:
        raise ValueError(f"Unknown case: {selected}")
    relative = f"images/{case['dataset']}__{case['imageId']}.jpg"
    raw = (EVIDENCE / relative).read_bytes()
    descriptor = manifest["files"][relative]
    digest = hashlib.sha256(raw).hexdigest()
    if digest != descriptor["sha256"] or len(raw) != descriptor["bytes"]:
        raise ValueError(f"Input image digest mismatch: {relative}")
    if args.show:
        print(
            json.dumps(
                {
                    "model": MODEL,
                    "revision": REVISION,
                    "labels": case["labels"],
                    "score_threshold": 0.1,
                    "image": {"path": relative, **descriptor},
                },
                indent=2,
            )
        )
        return 0
    key = os.environ.get("SIE_API_KEY")
    endpoint = os.environ.get("SIE_BASE_URL", "https://api.superlinked.com")
    if endpoint == "https://api.superlinked.com" and not key:
        raise ValueError("Set SIE_API_KEY, or use --show to inspect the request without sending it")
    from sie_sdk import SIEClient

    fmt = "png" if raw[:4] == b"\x89PNG" else "jpeg"
    with SIEClient(endpoint, api_key=key, timeout_s=120) as client:
        started = time.perf_counter()
        response = client.extract(MODEL, {"images": [{"data": raw, "format": fmt}]}, labels=case["labels"])
        elapsed = time.perf_counter() - started
        if client.last_model_revision != REVISION:
            raise ValueError("The served checkpoint differs from the recorded study")
    print(
        json.dumps(
            {
                "case": selected,
                "model": MODEL,
                "revision": REVISION,
                "seconds": elapsed,
                "response": {k: v for k, v in response.items() if k != "request"},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
