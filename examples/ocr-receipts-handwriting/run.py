#!/usr/bin/env python3
"""Read receipt photos and handwritten notes with SIE's LightOnOCR-2-1B.

    export SIE_API_KEY=sk-sie-...
    uv run run.py --images 20              # 20 CORD receipt photos, under $0.05
    uv run run.py --sets cord,gnhk --all   # all 272 photos the page may show, under $0.60
    python3 score.py --rescore runs

Each photo is converted the way the recorded run converted it (RGB, longest side at most
2,048 px, JPEG quality 92) and sent as one `extract` call. The text lands in
runs/outputs.jsonl, and the reference words in runs/references.jsonl, so score.py scores
your run with the rule the recorded one used.

The photos come from their publishers at pinned revisions: CORD v2's test split (NAVER
Clova, CC BY 4.0) and GNHK's test split (GoodNotes, CC BY 4.0, from a public mirror of the
original archive; about 1 GB, downloaded only with `--sets gnhk`). The SROIE receipts the
recorded run also read are not redistributed; their per-image counts are in the evidence.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODEL = "lightonai/LightOnOCR-2-1B"
CORD = ("naver-clova-ix/cord-v2", "7f0115a4b758a71d6473b8d085751692da2fef98")
GNHK = ("staghado/GNHK-Dataset", "eaa67d396fd29b8b04e38d630da79f67db11b212", "GNHK-dataset.zip")
MAX_SIDE = 2048


def jpeg(image) -> bytes:
    from PIL import Image

    image = image.convert("RGB")
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1.0:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def cord_photos() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    repo, revision = CORD
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision) if "/test-" in f]
    rows = []
    for name in sorted(files):
        table = pq.read_table(hf_hub_download(repo, name, repo_type="dataset", revision=revision)).to_pylist()
        for example in table:
            i = len(rows)
            truth = json.loads(example["ground_truth"])
            words = [w["text"] for line in truth.get("valid_line", []) for w in line.get("words", []) if w.get("text")]
            image = Image.open(io.BytesIO(example["image"]["bytes"]))
            rows.append({"id": f"cord/{i:04d}", "set": "cord", "words": words, "image": jpeg(image)})
    return rows


def gnhk_photos() -> list[dict]:
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageOps

    repo, revision, name = GNHK
    rows = []
    with zipfile.ZipFile(hf_hub_download(repo, name, repo_type="dataset", revision=revision)) as outer:
        members: dict[str, bytes] = {}
        for entry in outer.namelist():
            if entry.lower().endswith(".zip") and "test" in entry.lower():
                with zipfile.ZipFile(io.BytesIO(outer.read(entry))) as inner:
                    members.update({m: inner.read(m) for m in inner.namelist()})
            elif "test" in entry.lower() and entry.lower().endswith((".jpg", ".jpeg", ".png", ".json")):
                members[entry] = outer.read(entry)
        for entry in sorted(m for m in members if m.lower().endswith((".jpg", ".jpeg", ".png"))):
            stem = re.sub(r"\.(jpe?g|png)$", "", entry, flags=re.IGNORECASE)
            annotation = members.get(stem + ".json")
            if annotation is None or "__MACOSX" in entry:
                continue
            words = [
                w["text"]
                for w in json.loads(annotation)
                if isinstance(w.get("text"), str) and not re.fullmatch(r"%[^%]*%", w["text"].strip())
            ]
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(members[entry])))
            rows.append({"id": f"gnhk/{Path(stem).name}", "set": "gnhk", "words": words, "image": jpeg(image)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", default="cord", help="comma-separated: cord, gnhk")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=int, help="read this many photos per set")
    group.add_argument("--all", action="store_true", help="read every photo in each set")
    parser.add_argument("-c", "--concurrency", type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get("SIE_API_KEY"):
        print("Set SIE_API_KEY first (https://superlinked.com/cloud).", file=sys.stderr)
        return 1

    from sie_sdk import SIEClient

    loaders = {"cord": cord_photos, "gnhk": gnhk_photos}
    photos: list[dict] = []
    for name in args.sets.split(","):
        rows = loaders[name.strip()]()
        photos += rows if args.all else rows[: args.images]
    client = SIEClient(api_key=os.environ["SIE_API_KEY"], base_url="https://api.superlinked.com", timeout_s=900)

    def read(photo: dict) -> dict:
        sent = time.perf_counter()
        result = client.extract(MODEL, {"images": [{"data": photo["image"], "format": "jpeg"}]})
        text = "\n".join(e["text"] for e in result.get("entities", []))
        return {"id": photo["id"], "text": text, "seconds": round(time.perf_counter() - sent, 3)}

    RUNS.mkdir(exist_ok=True)
    outputs = []
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(read, photo) for photo in photos]
        for n, future in enumerate(as_completed(futures), 1):
            outputs.append(future.result())
            if n % 20 == 0 or n == len(photos):
                print(f"{n}/{len(photos)}")
    (RUNS / "outputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in outputs))
    references = [{"id": p["id"], "set": p["set"], "words": p["words"]} for p in photos]
    (RUNS / "references.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in references))
    print(f"Wrote {RUNS}. Now run: python3 score.py --rescore runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
