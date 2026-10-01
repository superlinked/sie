#!/usr/bin/env python3
"""Rebuild the frozen photos from their public sources, verifying every input hash."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from fetch import EVIDENCE, HERE
from score import load_evidence

RUNS = HERE / "runs"


def jpeg(image: Image.Image) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > 2048:
        scale = 2048 / max(image.size)
        image = image.resize(tuple(round(side * scale) for side in image.size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92)
    return out.getvalue()


def save(row: dict, image: Image.Image, words: list[str], source_bytes: bytes | None) -> dict:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    if pixel_sha != row["pixel_sha256"]:
        raise ValueError(f"{row['id']}: source pixels differ from the frozen input")
    if source_bytes is not None and hashlib.sha256(source_bytes).hexdigest() != row["source_sha256"]:
        raise ValueError(f"{row['id']}: source bytes differ from the frozen input")
    data = jpeg(rgb)
    if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
        raise ValueError(f"{row['id']}: JPEG encoder produced different bytes; no substituted input is accepted")
    target = RUNS / "images" / f"{row['id']}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {
        "id": row["id"],
        "set": row["set"],
        "image": str(target),
        "words": words,
        "image_sha256": row["image_sha256"],
    }


def main() -> int:
    manifest = load_evidence()
    samples = json.loads((EVIDENCE / "samples.json").read_text())
    if len(samples) != 96 or len({row["id"] for row in samples}) != 96:
        raise ValueError("The frozen comparison must contain 96 unique samples.")
    rows = []
    cord = load_dataset("naver-clova-ix/cord-v2", revision=manifest["cord_revision"], split="validation", token=False)
    for row in samples:
        if row["set"] != "cord":
            continue
        source = cord[int(row["id"].split("validation-")[1])]
        words = [
            word["text"]
            for line in json.loads(source["ground_truth"]).get("valid_line", [])
            for word in line.get("words", [])
            if word.get("text")
        ]
        rows.append(save(row, source["image"], words, None))
    archive = hf_hub_download(
        "staghado/GNHK-Dataset",
        "GNHK-dataset.zip",
        repo_type="dataset",
        revision=manifest["gnhk_revision"],
        token=False,
    )
    with zipfile.ZipFile(archive) as outer:
        nested_names = [name for name in outer.namelist() if name.lower().endswith(".zip") and "train" in name.lower()]
        if len(nested_names) != 1:
            raise ValueError("Expected one GNHK training archive.")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested_names[0]))) as train:
            photos = {
                Path(name).stem: name
                for name in train.namelist()
                if name.lower().endswith((".jpg", ".jpeg", ".png")) and "__MACOSX" not in name
            }
            for row in samples:
                if row["set"] != "gnhk":
                    continue
                stem = row["id"].split("gnhk/train-")[1]
                name = photos[stem]
                labels = json.loads(train.read(str(Path(name).with_suffix(".json"))))
                words = [
                    word["text"].strip()
                    for word in labels
                    if word.get("text", "").strip() and not re.fullmatch(r"%[^%]*%", word["text"].strip())
                ]
                data = train.read(name)
                rows.append(save(row, Image.open(io.BytesIO(data)), words, data))
    rows.sort(key=lambda row: row["id"])
    target = RUNS / "references.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Rebuilt and hash-verified {len(rows)} frozen inputs in {RUNS}; no model calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
