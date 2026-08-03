from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from sie_sdk import SIEClient
from sie_sdk.types import Item

from retail_shelf_audit.config import (
    DETECTION_LABELS,
    DETECTION_OPTIONS,
    DINO_MODEL,
    OCR_MODEL,
    ROOT,
    RUNS_DIR,
    SOURCE_IMAGE,
    load_config,
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _objects(result: dict[str, Any]) -> list[dict[str, Any]]:
    objects = result.get("objects", [])
    if not isinstance(objects, list):
        raise TypeError("Detector response has no object list")
    return [item for item in objects if isinstance(item, dict)]


def _ocr_text(result: dict[str, Any]) -> str:
    entities = result.get("entities", [])
    for entity in entities:
        if isinstance(entity, dict) and entity.get("label") == "markdown":
            return str(entity.get("text", ""))
    raise ValueError("OCR response has no markdown entity")


def _clean_detection(item: dict[str, Any]) -> dict[str, Any]:
    box = item.get("bbox", item.get("bbox_xywh"))
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Detection must contain a four-value bbox")
    return {
        "label": str(item["label"]),
        "score": round(float(item["score"]), 6),
        "bbox_xywh": [round(float(value), 1) for value in box],
    }


def select_gap(objects: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    image_width, image_height = image_size
    candidates: list[dict[str, Any]] = []
    for item in objects:
        box = item.get("bbox")
        if item.get("label") != "empty shelf space" or not isinstance(box, list) or len(box) != 4:
            continue
        _, _, width, height = (float(value) for value in box)
        if height <= 0 or width <= 0:
            continue
        is_full_width_strip = width >= image_width * 0.8 or width / height > 8
        is_too_short = height < image_height * 0.05
        if not is_full_width_strip and not is_too_short:
            candidates.append(item)
    if not candidates:
        raise ValueError("No non-strip empty shelf detection passed the geometry guard")
    return max(candidates, key=lambda item: float(item["score"]))


def _horizontal_overlap(first: list[float], second: list[float]) -> float:
    first_left, _, first_width, _ = first
    second_left, _, second_width, _ = second
    overlap = max(0.0, min(first_left + first_width, second_left + second_width) - max(first_left, second_left))
    return overlap / min(first_width, second_width)


def _iou(first: list[float], second: list[float]) -> float:
    first_left, first_top, first_width, first_height = first
    second_left, second_top, second_width, second_height = second
    left = max(first_left, second_left)
    top = max(first_top, second_top)
    right = min(first_left + first_width, second_left + second_width)
    bottom = min(first_top + first_height, second_top + second_height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def nearby_price_candidates(objects: list[dict[str, Any]], gap: dict[str, Any]) -> list[dict[str, Any]]:
    gap_box = [float(value) for value in gap["bbox"]]
    _, gap_top, _, gap_height = gap_box
    gap_bottom = gap_top + gap_height
    nearby: list[dict[str, Any]] = []
    for item in sorted(objects, key=lambda row: float(row.get("score", 0)), reverse=True):
        box = item.get("bbox")
        if item.get("label") != "price tag" or not isinstance(box, list) or len(box) != 4:
            continue
        numeric_box = [float(value) for value in box]
        _, top, _, height = numeric_box
        bottom = top + height
        within_vertical_band = top <= gap_bottom + gap_height and bottom >= gap_top - gap_height * 0.25
        if not within_vertical_band or _horizontal_overlap(gap_box, numeric_box) < 0.25:
            continue
        if any(_iou(numeric_box, [float(value) for value in kept["bbox"]]) >= 0.7 for kept in nearby):
            continue
        nearby.append(item)
    if len(nearby) < 2:
        raise ValueError("Fewer than two distinct nearby price-tag candidates passed the geometry guard")
    return nearby[:6]


def select_vertical_pair(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for first_index, first in enumerate(candidates):
        first_box = [float(value) for value in first["bbox"]]
        for second in candidates[first_index + 1 :]:
            second_box = [float(value) for value in second["bbox"]]
            overlap = _horizontal_overlap(first_box, second_box)
            if overlap < 0.6:
                continue
            upper, lower = sorted((first, second), key=lambda item: float(item["bbox"][1]))
            upper_box = [float(value) for value in upper["bbox"]]
            lower_box = [float(value) for value in lower["bbox"]]
            vertical_distance = abs(lower_box[1] - (upper_box[1] + upper_box[3]))
            score = float(upper["score"]) + float(lower["score"])
            pairs.append((overlap, score - vertical_distance / 10_000, upper, lower))
    if not pairs:
        raise ValueError("No vertically aligned DINO candidate pair passed the geometry guard")
    _, _, upper, lower = max(pairs, key=lambda row: (row[0], row[1]))
    return upper, lower


def candidate_crop_box(box_xywh: list[float], image_size: tuple[int, int]) -> list[int]:
    image_width, image_height = image_size
    x, y, width, height = (float(value) for value in box_xywh)
    x_pad = width * 0.2
    y_pad = height * 0.25
    return [
        max(0, round(x - x_pad)),
        max(0, round(y - y_pad)),
        min(image_width, round(x + width + x_pad)),
        min(image_height, round(y + height + y_pad)),
    ]


def create_candidate_crops(
    source: Path,
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        for index, detection in enumerate(candidates, start=1):
            crop_box = candidate_crop_box(detection["bbox"], rgb.size)
            crop = rgb.crop(tuple(crop_box))
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)
            path = output_dir / f"candidate-{index}.jpg"
            crop.save(path, format="JPEG", quality=95)
            records.append(
                {
                    "candidate_id": f"candidate-{index}",
                    "source_detection": _clean_detection(detection),
                    "source_crop_xyxy": crop_box,
                    "display_scale": 3,
                    "path": path,
                }
            )
    return records


def _unique_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and line not in lines:
            lines.append(line)
    return lines


def ocr_fragments(upper_text: str, lower_text: str) -> list[str]:
    upper_lines = _unique_lines(upper_text)
    lower_lines = [line for line in _unique_lines(lower_text) if line not in set(upper_lines)]
    if len(upper_lines) < 3 or len(lower_lines) < 4:
        raise ValueError("OCR did not return three upper lines and four distinct lower lines")
    return lower_lines[:4] + upper_lines[:3]


def build_evidence(
    gap_detection: dict[str, Any],
    label_detection: dict[str, Any],
    sign_text: str,
    price_text: str,
) -> dict[str, Any]:
    return {
        "gap_detection": _clean_detection(gap_detection),
        "price_detection": _clean_detection(label_detection),
        "ocr_fragments": ocr_fragments(sign_text, price_text),
    }


def _timed(call: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, round((time.perf_counter() - started) * 1000, 1)


def run_audit(run_id: str) -> Path:
    config = load_config()
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    crops_dir = run_dir / "crops"
    raw_dir.mkdir(parents=True, exist_ok=False)

    with Image.open(SOURCE_IMAGE) as image:
        image_size = image.size

    with SIEClient(
        config.base_url,
        api_key=config.api_key or None,
        timeout_s=config.request_timeout_s,
    ) as client:
        detector_result, detector_ms = _timed(
            lambda: client.extract(
                DINO_MODEL,
                Item(id="hitl-042", images=[SOURCE_IMAGE]),
                labels=DETECTION_LABELS,
                options=DETECTION_OPTIONS,
                wait_for_capacity=True,
                provision_timeout_s=config.provision_timeout_s,
            )
        )
        write_json(raw_dir / "grounding-dino.json", detector_result)

        objects = _objects(detector_result)
        gap = select_gap(objects, image_size)
        upper_detection, lower_detection = select_vertical_pair(nearby_price_candidates(objects, gap))
        crop_records = create_candidate_crops(SOURCE_IMAGE, [upper_detection, lower_detection], crops_dir)
        ocr_timings: dict[str, float] = {}
        for record in crop_records:
            result, duration_ms = _timed(
                lambda record=record: client.extract(
                    OCR_MODEL,
                    Item(id=record["candidate_id"], images=[record["path"]]),
                    options={"max_new_tokens": 64, "num_beams": 1},
                    wait_for_capacity=True,
                    provision_timeout_s=config.provision_timeout_s,
                )
            )
            write_json(raw_dir / f"lighton-ocr-{record['candidate_id']}.json", result)
            record["text"] = _ocr_text(result)
            ocr_timings[record["candidate_id"]] = duration_ms

        upper_record, lower_record = crop_records
        evidence = build_evidence(
            gap,
            lower_record["source_detection"],
            upper_record["text"],
            lower_record["text"],
        )
        write_json(run_dir / "evidence.json", evidence)
        write_json(
            run_dir / "selection.json",
            {
                "strategy": "non-strip gap, nearby price-tag candidates, vertically aligned upper/lower pair",
                "gap_detection": _clean_detection(gap),
                "upper_candidate_role": "notice",
                "upper_candidate_id": upper_record["candidate_id"],
                "lower_candidate_role": "shelf_label",
                "lower_candidate_id": lower_record["candidate_id"],
                "candidates": [
                    {
                        "candidate_id": record["candidate_id"],
                        "source_detection": record["source_detection"],
                        "source_crop_xyxy": record["source_crop_xyxy"],
                        "display_scale": record["display_scale"],
                        "ocr_text": record["text"],
                    }
                    for record in crop_records
                ],
            },
        )
    output_paths = sorted(
        (*raw_dir.glob("*.json"), *crops_dir.glob("*.jpg"), run_dir / "selection.json", run_dir / "evidence.json")
    )
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "endpoint": config.base_url,
            "execution": "SIE API",
            "checksum_scope": "source input plus every generated evidence file except this manifest",
            "source_input": {
                "path": SOURCE_IMAGE.relative_to(ROOT).as_posix(),
                "sha256": sha256(SOURCE_IMAGE),
            },
            "outputs": [
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "sha256": sha256(path),
                }
                for path in output_paths
            ],
            "timing_ms": {
                "grounding_dino": detector_ms,
                "lighton_ocr_candidates": ocr_timings,
            },
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and OCR the stockout evidence in the HITL 042 shelf image")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("cloud-%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    path = run_audit(args.run_id)
    print(path)
