"""Read and verify the immutable public Image Search recording."""

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET = "superlinked/sie-task-evidence"
REVISION = "3ca626712b086958963cfa5741b3993e1d81614e"
PREFIX = "evaluations/2026-10-06/image-search-0fe605cb2953dd92"
ARCHIVE_SHA256 = "7627df54b1fdbca000c33e29ece5942e05b618541e2d5c09145f32ca601141d8"
MANIFEST_SHA256 = "00d18b9ef00043510231f29acc14c981f4828d751591caa65fa1f7be1da63e91"
COST_SHA256 = "95075f104218fd9ce973c26d038565922effbea59ce8c90d59ec05c9afab3e66"
ROOT_NAME = "image20-8-semantic"
DEFAULT_ROOT = Path(__file__).resolve().parent / "evidence" / ROOT_NAME
ARM_DIMENSIONS = {
    "sie-siglip384": 1152,
    "sie-siglip2-base224": 768,
    "voyage35-default": 1024,
    "luna-caption-small": 1536,
    "sol61-caption-large": 3072,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(root: Path, name: str) -> Any:
    return json.loads((root / name).read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    manifest_bytes = (root / "manifest.json").read_bytes()
    if digest(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("Recording manifest differs from the pinned public manifest")
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        path = root / row["path"]
        data = path.read_bytes()
        if path.is_symlink() or len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError(f"Recording file failed its byte and SHA256 check: {row['path']}")
    cost_bytes = (root / "cost-projection.json").read_bytes()
    if digest(cost_bytes) != COST_SHA256:
        raise ValueError("Cost projection differs from the pinned corrected public companion")
    cost = json.loads(cost_bytes)
    if cost["source_archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Cost projection belongs to a different recording")
    return cost


def inputs(root: Path) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    images = read_json(root, "inputs/original-context/image-inputs.json")
    queries = read_json(root, "inputs/original-context/queries.json")
    gold = read_json(root, "inputs/original-context/gold.json")
    if not isinstance(images, list) or not isinstance(queries, list) or not isinstance(gold, dict):
        raise TypeError("Recording input structure is invalid")
    image_ids = [row["image_id"] for row in images]
    query_ids = [row["query_id"] for row in queries]
    if len(set(image_ids)) != 20 or len(images) != 20 or len(set(query_ids)) != 8 or len(queries) != 8:
        raise ValueError("The pinned pilot requires exactly 20 unique images and 8 unique queries")
    grades = {}
    for row in gold["rows"]:
        query_id = row["query_id"]
        if query_id in grades or query_id not in query_ids:
            raise ValueError("Duplicate or unknown gold query")
        relevance = row["relevance"]
        by_image = {item["image_id"]: item["grade"] for item in relevance}
        if len(relevance) != 20 or set(by_image) != set(image_ids):
            raise ValueError("Gold must grade every candidate exactly once")
        if any(type(grade) is not int or grade not in (0, 1, 2) for grade in by_image.values()):
            raise ValueError("Gold contains an ambiguous or invalid relevance grade")
        grades[query_id] = by_image
    if set(grades) != set(query_ids):
        raise ValueError("Gold must cover every query")
    return images, queries, grades
