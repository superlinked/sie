from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from retail_shelf_audit.audit import (
    build_evidence,
    candidate_crop_box,
    nearby_price_candidates,
    select_gap,
    select_vertical_pair,
)
from retail_shelf_audit.config import RECORDED_DIR, ROOT

EXPECTED_OCR_FRAGMENTS = [
    "Panadol Child",
    "5-12Yrs Elixir 100ml",
    "101760",
    "10⁹⁹",
    "I am temporarily",
    "out-of-stock",
    "from our supplier",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> None:
    manifest = _load(ROOT / "source-manifest.json")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {record['path']}: {actual}")
        if "width" in record or "height" in record:
            with Image.open(path) as image:
                if image.size != (record["width"], record["height"]):
                    raise ValueError(f"Dimension mismatch for {record['path']}: {image.size}")


def _recorded_ocr_pair(
    upper: dict[str, Any],
    lower: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ocr_run = _load(RECORDED_DIR / "lighton-ocr-derived-results.json")
    by_id = {row["case_id"]: row for row in ocr_run["cases"]}
    expected = [
        ("upper-notice", "upper_by_geometry", upper),
        ("lower-shelf-label", "lower_by_geometry", lower),
    ]
    for case_id, role, detection in expected:
        record = by_id[case_id]
        expected_detection = {
            "label": detection["label"],
            "score": round(float(detection["score"]), 6),
            "bbox_xywh": [round(float(value), 1) for value in detection["bbox"]],
        }
        expected_crop = candidate_crop_box(detection["bbox"], (4032, 3024))
        expected_size = (
            (expected_crop[2] - expected_crop[0]) * record["display_scale"],
            (expected_crop[3] - expected_crop[1]) * record["display_scale"],
        )
        crop_path = RECORDED_DIR / "derived-crops" / record["input_path"]
        if record["selection_role"] != role:
            raise ValueError(f"Recorded OCR role differs for {case_id}")
        if record["source_detection"] != expected_detection:
            raise ValueError(f"Recorded OCR source detection differs for {case_id}")
        if record["source_crop_xyxy"] != expected_crop:
            raise ValueError(f"Recorded OCR crop coordinates differ for {case_id}")
        if (record["image_width"], record["image_height"]) != expected_size:
            raise ValueError(f"Recorded OCR crop dimensions differ for {case_id}")
        if sha256(crop_path) != record["input_sha256"]:
            raise ValueError(f"Recorded OCR input checksum differs for {case_id}")
    return by_id["upper-notice"], by_id["lower-shelf-label"]


def recorded_evidence() -> dict[str, Any]:
    detector_run = _load(RECORDED_DIR / "launch-model-results.json")
    detections = detector_run["calls"]["detection"]["detections"]["042"]
    objects = [{"label": row["label"], "score": row["score"], "bbox": row["bbox_xywh"]} for row in detections]
    gap = select_gap(objects, (4032, 3024))
    upper, price = select_vertical_pair(nearby_price_candidates(objects, gap))
    upper_ocr, lower_ocr = _recorded_ocr_pair(upper, price)
    return build_evidence(
        gap,
        price,
        upper_ocr["text"],
        lower_ocr["text"],
    )


def verify_recorded_case() -> None:
    evidence = recorded_evidence()
    if evidence["ocr_fragments"] != EXPECTED_OCR_FRAGMENTS:
        raise ValueError("Recorded OCR fragments differ from the reviewed case-042 fixture")
    expected_detections = {
        "gap_detection": {
            "label": "empty shelf space",
            "score": 0.274157,
            "bbox_xywh": [2043.8, 2137.0, 623.6, 402.4],
        },
        "price_detection": {
            "label": "price tag",
            "score": 0.259638,
            "bbox_xywh": [2235.5, 2559.4, 399.2, 186.3],
        },
    }
    mismatches = {
        key: (evidence.get(key), value) for key, value in expected_detections.items() if evidence.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Recorded detector evidence differs from reviewed case 042: {mismatches}")


def main() -> None:
    verify_manifest()
    verify_recorded_case()
    print("Recorded checksums, DINO geometry, and derived-crop OCR evidence are valid.")
